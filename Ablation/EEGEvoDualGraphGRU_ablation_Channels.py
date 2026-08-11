import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

CGX_CHANNEL_NAMES = [
    "F7", "Fp1", "Fp2", "F8", "F3", "Fz", "F4", "C3", "Cz",
    "P8", "P7", "Pz", "P4", "T3", "P3", "O1", "O2", "C4", "T4"
]


def calculate_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    model_size_bytes = total_params * 4
    model_size_mb = model_size_bytes / (1024 ** 2)
    return model_size_mb, total_params


CHANNEL_POS = {
    "Fp1": (-2.0, 4.0), "Fp2": (2.0, 4.0),
    "F7": (-4.0, 2.0), "F3": (-2.0, 2.0), "Fz": (0.0, 2.2), "F4": (2.0, 2.0), "F8": (4.0, 2.0),
    "T3": (-4.5, 0.0), "C3": (-2.0, 0.0), "Cz": (0.0, 0.0), "C4": (2.0, 0.0), "T4": (4.5, 0.0),
    "P7": (-4.0, -2.0), "P3": (-2.0, -2.0), "Pz": (0.0, -2.2), "P4": (2.0, -2.0), "P8": (4.0, -2.0),
    "O1": (-2.0, -4.0), "O2": (2.0, -4.0),
}


def build_geometry_bias(channel_names, sigma=2.0, self_loop=True):
    coords = torch.tensor([CHANNEL_POS[ch] for ch in channel_names], dtype=torch.float32)
    dist = torch.cdist(coords, coords, p=2)
    bias = torch.exp(-(dist ** 2) / (2 * sigma ** 2))
    if self_loop:
        bias.fill_diagonal_(1.0)
    return bias


def dense_adj_to_edge_index_and_weight(adj, eps=1e-8):
    idx = (adj > eps).nonzero(as_tuple=False).t().contiguous()
    w = adj[idx[0], idx[1]]
    return idx, w


def build_batched_edge_index_from_adjs(adj_batch, eps=1e-8):
    device = adj_batch.device
    B, N, _ = adj_batch.shape
    edge_indices = []
    edge_weights = []
    for b in range(B):
        edge_index, edge_weight = dense_adj_to_edge_index_and_weight(adj_batch[b], eps=eps)
        edge_index = edge_index + b * N
        edge_indices.append(edge_index)
        edge_weights.append(edge_weight)
    edge_index_b = torch.cat(edge_indices, dim=1).to(device)
    edge_weight_b = torch.cat(edge_weights, dim=0).to(device)
    return edge_index_b, edge_weight_b


def topk_rowwise(adj, k):
    if k is None or k >= adj.size(-1):
        return adj
    vals, idx = torch.topk(adj, k=k, dim=-1)
    mask = torch.zeros_like(adj)
    mask.scatter_(-1, idx, 1.0)
    adj = adj * mask
    adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return adj


class MultiScaleTemporalStem(nn.Module):
    def __init__(self, input_dim, branch_dim=4, kernels=(3, 7), dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.branch_dim = branch_dim
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, input_dim * branch_dim, kernel_size=k, padding=k // 2, groups=input_dim, bias=False),
                nn.BatchNorm1d(input_dim * branch_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for k in kernels
        ])
        self.out_dim = len(kernels) * branch_dim

    def forward(self, x):
        x_c = x.transpose(1, 2)
        outs = []
        for branch in self.branches:
            y = branch(x_c)
            B, CD, T = y.shape
            y = y.view(B, self.input_dim, self.branch_dim, T).permute(0, 3, 1, 2)
            outs.append(y)
        return torch.cat(outs, dim=-1)


class DynamicLocalBandPowerEncoder(nn.Module):
    def __init__(self, sfreq=100, bands=((4, 8), (8, 13), (13, 30), (30, 45)), window_size=20, use_relative=True, log_power=True, eps=1e-6):
        super().__init__()
        self.sfreq = sfreq
        self.bands = bands
        self.window_size = window_size
        self.use_relative = use_relative
        self.log_power = log_power
        self.eps = eps

    def forward(self, x):
        B, T, C = x.shape
        x_bc_t = x.transpose(1, 2)
        pad_left = self.window_size // 2
        pad_right = self.window_size - 1 - pad_left
        x_bc_t = F.pad(x_bc_t, (pad_left, pad_right), mode="reflect")
        x_win = x_bc_t.unfold(dimension=-1, size=self.window_size, step=1)
        win = torch.hann_window(self.window_size, device=x.device, dtype=x.dtype).view(1, 1, 1, self.window_size)
        x_win = x_win * win
        fft = torch.fft.rfft(x_win, dim=-1)
        psd = (fft.real ** 2 + fft.imag ** 2) / self.window_size
        freqs = torch.fft.rfftfreq(self.window_size, d=1.0 / self.sfreq).to(x.device)

        total_mask = (freqs >= self.bands[0][0]) & (freqs < self.bands[-1][1])
        total_power = psd[..., total_mask].sum(dim=-1, keepdim=True) + self.eps
        band_feats = []
        for f_low, f_high in self.bands:
            mask = (freqs >= f_low) & (freqs < f_high)
            band_feats.append(psd[..., mask].sum(dim=-1))
        band_feats = torch.stack(band_feats, dim=-1)
        abs_feats = torch.log(band_feats + self.eps) if self.log_power else band_feats
        if self.use_relative:
            rel_feats = band_feats / total_power
            rel_feats = torch.log(rel_feats + self.eps) if self.log_power else rel_feats
            out = torch.cat([abs_feats, rel_feats], dim=-1)
        else:
            out = abs_feats
        return out.permute(0, 2, 1, 3).contiguous()


class WindowAggregator(nn.Module):
    def __init__(self, seq_len=100, num_windows=5):
        super().__init__()
        assert seq_len % num_windows == 0
        self.seq_len = seq_len
        self.num_windows = num_windows
        self.win_len = seq_len // num_windows

    def forward(self, x):
        B, T, C, D = x.shape
        return x.view(B, self.num_windows, self.win_len, C, D).mean(dim=2)


class PairwiseEdgeScorer(nn.Module):
    def __init__(self, d_model, hidden_dim=64):
        super().__init__()
        self.src_proj = nn.Linear(d_model, hidden_dim)
        self.dst_proj = nn.Linear(d_model, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, geom_bias):
        B, C, _ = x.shape
        src = self.src_proj(x)
        dst = self.dst_proj(x)
        si = src.unsqueeze(2).expand(B, C, C, -1)
        dj = dst.unsqueeze(1).expand(B, C, C, -1)
        geom = geom_bias.unsqueeze(0).unsqueeze(-1).expand(B, C, C, 1)
        feat = torch.cat([si, dj, si - dj, si * dj, geom], dim=-1)
        return self.edge_mlp(feat).squeeze(-1)


class EvolvingSharedSpecificGraphLearner(nn.Module):
    def __init__(self, d_model, num_nodes, geom_bias, hidden_dim=64, topk=8, temperature=0.7, geom_mix=0.2, residual_scale=0.5):
        super().__init__()
        self.num_nodes = num_nodes
        self.topk = topk
        self.temperature = temperature
        self.geom_mix = geom_mix
        self.residual_scale = residual_scale
        self.temp_base_scorer = PairwiseEdgeScorer(d_model, hidden_dim)
        self.freq_base_scorer = PairwiseEdgeScorer(d_model, hidden_dim)
        self.temp_res_scorer = PairwiseEdgeScorer(d_model, hidden_dim)
        self.freq_res_scorer = PairwiseEdgeScorer(d_model, hidden_dim)
        self.state_gate = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.register_buffer("geom_bias", geom_bias.float())

    def _normalize_adj(self, scores):
        adj = torch.softmax(scores / self.temperature, dim=-1)
        adj = topk_rowwise(adj, self.topk)
        eye = torch.eye(self.num_nodes, device=adj.device).unsqueeze(0)
        adj = 0.9 * adj + 0.1 * eye
        return adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(self, x_t, x_f):
        B, W, C, _ = x_t.shape
        shared_list, temp_list, freq_list, delta_t_list, delta_f_list = [], [], [], [], []
        A_shared_prev = None
        geom = self.geom_bias.unsqueeze(0)
        for w in range(W):
            xt, xf = x_t[:, w], x_f[:, w]
            E_t_base = self.temp_base_scorer(xt, self.geom_bias)
            E_f_base = self.freq_base_scorer(xf, self.geom_bias)
            E_shared_raw = 0.5 * (E_t_base + E_f_base)
            E_shared_raw = (1.0 - self.geom_mix) * E_shared_raw + self.geom_mix * geom
            A_shared_candidate = self._normalize_adj(E_shared_raw)
            if A_shared_prev is None:
                A_shared = A_shared_candidate
            else:
                g_in = torch.cat([xt.mean(dim=1), xf.mean(dim=1)], dim=-1)
                alpha = self.state_gate(g_in).view(B, 1, 1)
                A_shared = alpha * A_shared_prev + (1.0 - alpha) * A_shared_candidate
                A_shared = A_shared / A_shared.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            E_dt = self.temp_res_scorer(xt, self.geom_bias)
            E_df = self.freq_res_scorer(xf, self.geom_bias)
            Delta_t = torch.tanh(E_dt) * self.residual_scale
            Delta_f = torch.tanh(E_df) * self.residual_scale
            A_t = self._normalize_adj(torch.log(A_shared.clamp_min(1e-8)) + Delta_t)
            A_f = self._normalize_adj(torch.log(A_shared.clamp_min(1e-8)) + Delta_f)
            shared_list.append(A_shared); temp_list.append(A_t); freq_list.append(A_f)
            delta_t_list.append(Delta_t); delta_f_list.append(Delta_f)
            A_shared_prev = A_shared
        return (
            torch.stack(shared_list, dim=1),
            torch.stack(temp_list, dim=1),
            torch.stack(freq_list, dim=1),
            torch.stack(delta_t_list, dim=1),
            torch.stack(delta_f_list, dim=1),
        )


class DynamicGraphBlockFromAdj(nn.Module):
    def __init__(self, d_model, num_nodes, dropout=0.3):
        super().__init__()
        self.conv = GCNConv(d_model, d_model, add_self_loops=False, normalize=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjs):
        B, W, C, D = x.shape
        outs = []
        for w in range(W):
            xw = x[:, w]
            Aw = adjs[:, w]
            edge_index_b, edge_weight_b = build_batched_edge_index_from_adjs(Aw)
            xw_flat = xw.reshape(B * C, D)
            residual = xw_flat
            out = self.conv(xw_flat, edge_index_b, edge_weight_b)
            out = F.gelu(out)
            out = self.dropout(out)
            out = self.norm(out)
            out = out + residual
            outs.append(out.view(B, C, D))
        return torch.stack(outs, dim=1)


class ChannelAttentionFusion(nn.Module):
    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        hidden = max(16, d_model // 2)
        self.score = nn.Sequential(nn.Linear(d_model, hidden), nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.value = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, x):
        s = self.score(x)
        w = torch.softmax(s, dim=2)
        v = self.value(x)
        g = self.gate(x)
        return (w * g * v).sum(dim=2)


class SequenceAttentionPooling(nn.Module):
    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        hidden = max(16, d_model // 2)
        self.pool = nn.Sequential(nn.Linear(d_model, hidden), nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        w = torch.softmax(self.pool(x), dim=1)
        return (x * w).sum(dim=1)


class BranchHead(nn.Module):
    def __init__(self, d_model, num_classes=2, dropout=0.3):
        super().__init__()
        self.channel_fusion = ChannelAttentionFusion(d_model, dropout)
        self.seq_pool = SequenceAttentionPooling(d_model, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        z = self.channel_fusion(x)
        z = self.seq_pool(z)
        return self.classifier(z), z


class EEGEvoDualGraphGRUAblation(nn.Module):
    def __init__(
        self,
        input_dim=19,
        seq_len=100,
        num_classes=2,
        sfreq=100,
        num_windows=10,
        temporal_branch_dim=8,
        node_dim=128,
        graph_hidden_dim=128,
        gru_hidden=128,
        gru_layers=3,
        dropout=0.3,
        bidirectional=True,
        graph_topk=8,
        graph_temperature=0.7,
        graph_geom_mix=0.2,
        residual_scale=0.5,
        freq_bands=((4, 8), (8, 13), (13, 30), (30, 45)),
        local_window_size=20,
        aux_loss_weight=0.3,
        channel_names=None,
        branch_mode="dual",
    ):
        super().__init__()
        if channel_names is None:
            channel_names = CGX_CHANNEL_NAMES[:input_dim]
        assert len(channel_names) == input_dim
        if branch_mode not in {"dual", "temporal_only", "frequency_only"}:
            raise ValueError(f"Unsupported branch_mode: {branch_mode}")

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.sfreq = sfreq
        self.num_windows = num_windows
        self.node_dim = node_dim
        self.aux_loss_weight = aux_loss_weight
        self.branch_mode = branch_mode
        self.channel_names = list(channel_names)

        geom_bias = build_geometry_bias(self.channel_names, sigma=2.0, self_loop=True)
        self.register_buffer("geom_bias", geom_bias)

        self.temporal_stem = MultiScaleTemporalStem(input_dim=input_dim, branch_dim=temporal_branch_dim, kernels=(3, 7), dropout=dropout)
        self.band_encoder = DynamicLocalBandPowerEncoder(sfreq=sfreq, bands=freq_bands, window_size=local_window_size, use_relative=True, log_power=True)

        temporal_in_dim = 1 + self.temporal_stem.out_dim
        freq_in_dim = len(freq_bands) * 2
        self.temporal_proj = nn.Sequential(nn.Linear(temporal_in_dim, node_dim), nn.GELU(), nn.Dropout(dropout))
        self.frequency_proj = nn.Sequential(nn.Linear(freq_in_dim, node_dim), nn.GELU(), nn.Dropout(dropout))
        self.temporal_window = WindowAggregator(seq_len=seq_len, num_windows=num_windows)
        self.frequency_window = WindowAggregator(seq_len=seq_len, num_windows=num_windows)

        self.graph_learner = EvolvingSharedSpecificGraphLearner(
            d_model=node_dim,
            num_nodes=input_dim,
            geom_bias=self.geom_bias,
            hidden_dim=graph_hidden_dim,
            topk=graph_topk,
            temperature=graph_temperature,
            geom_mix=graph_geom_mix,
            residual_scale=residual_scale,
        )
        self.temporal_graph = DynamicGraphBlockFromAdj(d_model=node_dim, num_nodes=input_dim, dropout=dropout)
        self.frequency_graph = DynamicGraphBlockFromAdj(d_model=node_dim, num_nodes=input_dim, dropout=dropout)
        self.temporal_aux_head = BranchHead(node_dim, num_classes=num_classes, dropout=dropout)
        self.frequency_aux_head = BranchHead(node_dim, num_classes=num_classes, dropout=dropout)
        self.channel_fusion = ChannelAttentionFusion(node_dim, dropout=dropout)
        self.seq_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=node_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        gru_out_dim = gru_hidden * (2 if bidirectional else 1)
        self.post_gru_dropout = nn.Dropout(dropout)
        self.seq_pool = SequenceAttentionPooling(gru_out_dim, dropout=dropout)
        self.classifier = nn.Sequential(nn.Linear(gru_out_dim, gru_out_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(gru_out_dim, num_classes))

    def _normalize(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-5)
        return (x - mean) / std

    def _build_temporal_nodes(self, x):
        raw = x.unsqueeze(-1)
        temp = self.temporal_stem(x)
        x = torch.cat([raw, temp], dim=-1)
        x = self.temporal_proj(x)
        return self.temporal_window(x)

    def _build_frequency_nodes(self, x):
        freq = self.band_encoder(x)
        freq = self.frequency_proj(freq)
        return self.frequency_window(freq)

    def _branch_stats_from_single_graph(self, A_graph):
        zeros = torch.zeros_like(A_graph)
        return {
            "A_shared": A_graph,
            "A_temp": A_graph,
            "A_freq": A_graph,
            "Delta_temp": zeros,
            "Delta_freq": zeros,
        }

    def forward(self, x, return_features=False):
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, C], got {x.shape}")
        if x.size(1) != self.seq_len or x.size(2) != self.input_dim:
            raise ValueError(f"Expected [B, {self.seq_len}, {self.input_dim}], got {x.shape}")

        x_norm = self._normalize(x)
        x_t_nodes = self._build_temporal_nodes(x_norm)
        x_f_nodes = self._build_frequency_nodes(x_norm)

        if self.branch_mode == "dual":
            A_shared, A_t, A_f, Delta_t, Delta_f = self.graph_learner(x_t_nodes, x_f_nodes)
            x_t_graph = self.temporal_graph(x_t_nodes, A_t)
            x_f_graph = self.frequency_graph(x_f_nodes, A_f)
            temp_logits, temp_embed = self.temporal_aux_head(x_t_graph)
            freq_logits, freq_embed = self.frequency_aux_head(x_f_graph)
            x_branch = 0.5 * (x_t_graph + x_f_graph)
        elif self.branch_mode == "temporal_only":
            A_t, _, _, _, _ = self.graph_learner(x_t_nodes, x_t_nodes)
            x_t_graph = self.temporal_graph(x_t_nodes, A_t)
            temp_logits, temp_embed = self.temporal_aux_head(x_t_graph)
            x_f_graph = None
            freq_logits = temp_logits.detach() * 0.0
            freq_embed = temp_embed.detach() * 0.0
            x_branch = x_t_graph
            A_shared = A_t
            A_f = A_t
            Delta_t = torch.zeros_like(A_t)
            Delta_f = torch.zeros_like(A_t)
        else:
            A_f, _, _, _, _ = self.graph_learner(x_f_nodes, x_f_nodes)
            x_f_graph = self.frequency_graph(x_f_nodes, A_f)
            freq_logits, freq_embed = self.frequency_aux_head(x_f_graph)
            x_t_graph = None
            temp_logits = freq_logits.detach() * 0.0
            temp_embed = freq_embed.detach() * 0.0
            x_branch = x_f_graph
            A_shared = A_f
            A_t = A_f
            Delta_t = torch.zeros_like(A_f)
            Delta_f = torch.zeros_like(A_f)

        x_seq = self.channel_fusion(x_branch)
        x_seq = self.seq_dropout(x_seq)
        x_seq, _ = self.gru(x_seq)
        x_seq = self.post_gru_dropout(x_seq)
        x_global = self.seq_pool(x_seq)
        fused_logits = self.classifier(x_global)

        out = {
            "fused_logits": fused_logits,
            "temp_logits": temp_logits,
            "freq_logits": freq_logits,
            "A_shared": A_shared,
            "A_temp": A_t,
            "A_freq": A_f,
            "Delta_temp": Delta_t,
            "Delta_freq": Delta_f,
        }
        if return_features:
            out.update({
                "normalized": x_norm,
                "temporal_nodes": x_t_nodes,
                "frequency_nodes": x_f_nodes,
                "temporal_graph": x_t_graph if x_t_graph is not None else x_branch,
                "frequency_graph": x_f_graph if x_f_graph is not None else x_branch,
                "dual_graph": x_branch,
                "temp_embed": temp_embed,
                "freq_embed": freq_embed,
                "seq_features": x_seq,
                "global_features": x_global,
            })
        return out


def graph_regularization_loss(outputs, geom_bias, smooth_weight=0.10, shared_weight=0.10, residual_weight=0.02, geom_weight=0.05):
    A_shared = outputs["A_shared"]
    A_temp = outputs["A_temp"]
    A_freq = outputs["A_freq"]
    Delta_temp = outputs["Delta_temp"]
    Delta_freq = outputs["Delta_freq"]
    loss_smooth = ((A_shared[:, 1:] - A_shared[:, :-1]) ** 2).mean() if A_shared.size(1) > 1 else A_shared.new_tensor(0.0)
    loss_shared = ((A_temp - A_shared) ** 2).mean() + ((A_freq - A_shared) ** 2).mean()
    loss_residual = (Delta_temp ** 2).mean() + (Delta_freq ** 2).mean()
    geom_penalty = (1.0 - geom_bias).unsqueeze(0).unsqueeze(0)
    loss_geom = (geom_penalty * (A_shared + A_temp + A_freq) / 3.0).mean()
    total_graph_reg = smooth_weight * loss_smooth + shared_weight * loss_shared + residual_weight * loss_residual + geom_weight * loss_geom
    stats = {
        "loss_graph_smooth": float(loss_smooth.detach().item()),
        "loss_graph_shared": float(loss_shared.detach().item()),
        "loss_graph_residual": float(loss_residual.detach().item()),
        "loss_graph_geom": float(loss_geom.detach().item()),
    }
    return total_graph_reg, stats


def evo_dual_graph_loss(model, outputs, targets, label_smoothing=0.05, aux_weight=0.3, smooth_weight=0.10, shared_weight=0.10, residual_weight=0.02, geom_weight=0.05):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    loss_fused = criterion(outputs["fused_logits"], targets)
    loss_temp = criterion(outputs["temp_logits"], targets)
    loss_freq = criterion(outputs["freq_logits"], targets)
    loss_graph_reg, graph_stats = graph_regularization_loss(outputs, model.geom_bias, smooth_weight, shared_weight, residual_weight, geom_weight)

    if model.branch_mode == "dual":
        loss = loss_fused + aux_weight * loss_temp + aux_weight * loss_freq + loss_graph_reg
    elif model.branch_mode == "temporal_only":
        loss = loss_fused + aux_weight * loss_temp + loss_graph_reg
        loss_freq = loss_fused.new_tensor(0.0)
    else:
        loss = loss_fused + aux_weight * loss_freq + loss_graph_reg
        loss_temp = loss_fused.new_tensor(0.0)

    stats = {
        "loss_total": float(loss.item()),
        "loss_fused": float(loss_fused.item()),
        "loss_temp": float(loss_temp.item()),
        "loss_freq": float(loss_freq.item()),
        "loss_graph_reg": float(loss_graph_reg.item()),
    }
    stats.update(graph_stats)
    return loss, stats
