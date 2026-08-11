import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


# ============================================================
# Utilities
# ============================================================

CGX_CHANNEL_NAMES = [
    "F7", "Fp1", "Fp2", "F8", "F3", "Fz", "F4", "C3", "Cz",
    "P8", "P7", "Pz", "P4", "T3", "P3", "O1", "O2", "C4", "T4"
]


def calculate_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    model_size_bytes = total_params * 4
    model_size_mb = model_size_bytes / (1024 ** 2)
    return model_size_mb, total_params


def build_geometry_bias(channel_names, sigma=2.0, self_loop=True):
    """
    Soft geometry prior from approximate 2D scalp coordinates.
    Returns [C, C].
    """
    channel_pos = {
        "Fp1": (-2.0,  4.0),
        "Fp2": ( 2.0,  4.0),

        "F7":  (-4.0,  2.0),
        "F3":  (-2.0,  2.0),
        "Fz":  ( 0.0,  2.2),
        "F4":  ( 2.0,  2.0),
        "F8":  ( 4.0,  2.0),

        "T3":  (-4.5,  0.0),
        "C3":  (-2.0,  0.0),
        "Cz":  ( 0.0,  0.0),
        "C4":  ( 2.0,  0.0),
        "T4":  ( 4.5,  0.0),

        "P7":  (-4.0, -2.0),
        "P3":  (-2.0, -2.0),
        "Pz":  ( 0.0, -2.2),
        "P4":  ( 2.0, -2.0),
        "P8":  ( 4.0, -2.0),

        "O1":  (-2.0, -4.0),
        "O2":  ( 2.0, -4.0),
    }

    coords = torch.tensor([channel_pos[ch] for ch in channel_names], dtype=torch.float32)
    dist = torch.cdist(coords, coords, p=2)
    bias = torch.exp(-(dist ** 2) / (2 * sigma ** 2))
    if self_loop:
        bias.fill_diagonal_(1.0)
    return bias


def dense_adj_to_edge_index_and_weight(adj, eps=1e-8):
    """
    adj: [N, N]
    returns edge_index [2, E], edge_weight [E]
    Keeps edges > eps
    """
    idx = (adj > eps).nonzero(as_tuple=False).t().contiguous()
    w = adj[idx[0], idx[1]]
    return idx, w


def build_batched_edge_index_from_adjs(adj_batch, eps=1e-8):
    """
    adj_batch: [B, N, N]
    Returns:
        edge_index_b: [2, E_total]
        edge_weight_b: [E_total]
    """
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
    """
    adj: [B, C, C]
    Keep top-k per row, then renormalize.
    """
    if k is None or k >= adj.size(-1):
        return adj

    vals, idx = torch.topk(adj, k=k, dim=-1)
    mask = torch.zeros_like(adj)
    mask.scatter_(-1, idx, 1.0)
    adj = adj * mask
    adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return adj


# ============================================================
# Feature encoders
# ============================================================

class MultiScaleTemporalStem(nn.Module):
    """
    Input : [B, T, C]
    Output: [B, T, C, D_t]
    """
    def __init__(self, input_dim, branch_dim=4, kernels=(3, 7), dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.branch_dim = branch_dim

        branches = []
        for k in kernels:
            branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=input_dim,
                        out_channels=input_dim * branch_dim,
                        kernel_size=k,
                        padding=k // 2,
                        groups=input_dim,
                        bias=False,
                    ),
                    nn.BatchNorm1d(input_dim * branch_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.out_dim = len(kernels) * branch_dim

    def forward(self, x):
        # x: [B, T, C]
        x_c = x.transpose(1, 2)  # [B, C, T]
        outs = []
        for branch in self.branches:
            y = branch(x_c)  # [B, C * branch_dim, T]
            B, CD, T = y.shape
            y = y.view(B, self.input_dim, self.branch_dim, T).permute(0, 3, 1, 2)
            outs.append(y)
        return torch.cat(outs, dim=-1)  # [B, T, C, D_t]


class DynamicLocalBandPowerEncoder(nn.Module):
    """
    Input : [B, T, C]
    Output: [B, T, C, F]
    """
    def __init__(
        self,
        sfreq=100,
        bands=((4, 8), (8, 13), (13, 30), (30, 45)),
        window_size=20,
        use_relative=True,
        log_power=True,
        eps=1e-6,
    ):
        super().__init__()
        self.sfreq = sfreq
        self.bands = bands
        self.window_size = window_size
        self.use_relative = use_relative
        self.log_power = log_power
        self.eps = eps

    def forward(self, x):
        B, T, C = x.shape
        x_bc_t = x.transpose(1, 2)  # [B, C, T]

        pad_left = self.window_size // 2
        pad_right = self.window_size - 1 - pad_left
        x_bc_t = F.pad(x_bc_t, (pad_left, pad_right), mode="reflect")

        x_win = x_bc_t.unfold(dimension=-1, size=self.window_size, step=1)  # [B, C, T, W]

        win = torch.hann_window(
            self.window_size,
            device=x.device,
            dtype=x.dtype
        ).view(1, 1, 1, self.window_size)

        x_win = x_win * win

        fft = torch.fft.rfft(x_win, dim=-1)
        psd = (fft.real ** 2 + fft.imag ** 2) / self.window_size
        freqs = torch.fft.rfftfreq(self.window_size, d=1.0 / self.sfreq).to(x.device)

        band_feats = []
        total_mask = (freqs >= self.bands[0][0]) & (freqs < self.bands[-1][1])
        total_power = psd[..., total_mask].sum(dim=-1, keepdim=True) + self.eps

        for f_low, f_high in self.bands:
            mask = (freqs >= f_low) & (freqs < f_high)
            bp = psd[..., mask].sum(dim=-1)
            band_feats.append(bp)

        band_feats = torch.stack(band_feats, dim=-1)  # [B, C, T, nbands]

        abs_feats = torch.log(band_feats + self.eps) if self.log_power else band_feats

        if self.use_relative:
            rel_feats = band_feats / total_power
            rel_feats = torch.log(rel_feats + self.eps) if self.log_power else rel_feats
            out = torch.cat([abs_feats, rel_feats], dim=-1)
        else:
            out = abs_feats

        return out.permute(0, 2, 1, 3).contiguous()  # [B, T, C, F]


class WindowAggregator(nn.Module):
    """
    Convert [B, T, C, D] -> [B, W, C, D] by average pooling over time windows.
    """
    def __init__(self, seq_len=100, num_windows=5):
        super().__init__()
        assert seq_len % num_windows == 0, "seq_len must be divisible by num_windows"
        self.seq_len = seq_len
        self.num_windows = num_windows
        self.win_len = seq_len // num_windows

    def forward(self, x):
        B, T, C, D = x.shape
        x = x.view(B, self.num_windows, self.win_len, C, D).mean(dim=2)
        return x  # [B, W, C, D]


# ============================================================
# Graph learning
# ============================================================

class PairwiseEdgeScorer(nn.Module):
    """
    Directed edge scorer:
        e_ij = MLP([src_i, dst_j, src_i-dst_j, src_i*dst_j, geom_ij])
    Produces asymmetric scores.
    """
    def __init__(self, d_model, hidden_dim=64):
        super().__init__()
        self.src_proj = nn.Linear(d_model, hidden_dim)
        self.dst_proj = nn.Linear(d_model, hidden_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, geom_bias):
        # x: [B, C, D]
        # geom_bias: [C, C]
        B, C, D = x.shape

        src = self.src_proj(x)  # [B, C, H]
        dst = self.dst_proj(x)  # [B, C, H]

        si = src.unsqueeze(2).expand(B, C, C, -1)
        dj = dst.unsqueeze(1).expand(B, C, C, -1)

        geom = geom_bias.unsqueeze(0).unsqueeze(-1).expand(B, C, C, 1)

        feat = torch.cat([
            si,
            dj,
            si - dj,
            si * dj,
            geom
        ], dim=-1)

        e = self.edge_mlp(feat).squeeze(-1)  # [B, C, C]
        return e


class EvolvingSharedSpecificGraphLearner(nn.Module):
    """
    Learn dynamic graphs across windows with shared-specific decomposition.

    For each window w:
        E_t_base, E_f_base -> shared edge base
        delta_t, delta_f   -> branch-specific residuals
        A_shared^w         -> temporally evolved graph state
        A_t^w, A_f^w       -> branch graphs

    Returns:
        A_shared_seq: [B, W, C, C]
        A_t_seq:      [B, W, C, C]
        A_f_seq:      [B, W, C, C]
        Delta_t_seq:  [B, W, C, C]
        Delta_f_seq:  [B, W, C, C]
    """
    def __init__(
        self,
        d_model,
        num_nodes,
        geom_bias,
        hidden_dim=64,
        topk=8,
        temperature=0.7,
        geom_mix=0.2,
        residual_scale=0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.topk = topk
        self.temperature = temperature
        self.geom_mix = geom_mix
        self.residual_scale = residual_scale

        self.temp_base_scorer = PairwiseEdgeScorer(d_model, hidden_dim=hidden_dim)
        self.freq_base_scorer = PairwiseEdgeScorer(d_model, hidden_dim=hidden_dim)

        self.temp_res_scorer = PairwiseEdgeScorer(d_model, hidden_dim=hidden_dim)
        self.freq_res_scorer = PairwiseEdgeScorer(d_model, hidden_dim=hidden_dim)

        self.state_gate = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.register_buffer("geom_bias", geom_bias.float())

    def _normalize_adj(self, scores):
        adj = torch.softmax(scores / self.temperature, dim=-1)
        adj = topk_rowwise(adj, self.topk)

        eye = torch.eye(self.num_nodes, device=adj.device).unsqueeze(0)
        adj = 0.9 * adj + 0.1 * eye
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return adj

    def forward(self, x_t, x_f):
        # x_t, x_f: [B, W, C, D]
        B, W, C, D = x_t.shape

        shared_list = []
        temp_list = []
        freq_list = []
        delta_t_list = []
        delta_f_list = []

        A_shared_prev = None

        geom = self.geom_bias.unsqueeze(0)  # [1, C, C]

        for w in range(W):
            xt = x_t[:, w]  # [B, C, D]
            xf = x_f[:, w]  # [B, C, D]

            E_t_base = self.temp_base_scorer(xt, self.geom_bias)
            E_f_base = self.freq_base_scorer(xf, self.geom_bias)

            E_shared_raw = 0.5 * (E_t_base + E_f_base)
            E_shared_raw = (1.0 - self.geom_mix) * E_shared_raw + self.geom_mix * geom

            A_shared_candidate = self._normalize_adj(E_shared_raw)

            if A_shared_prev is None:
                A_shared = A_shared_candidate
            else:
                # Graph-state evolution gate from both views
                g_in = torch.cat([xt.mean(dim=1), xf.mean(dim=1)], dim=-1)  # [B, 2D]
                alpha = self.state_gate(g_in).view(B, 1, 1)                 # [B,1,1]
                A_shared = alpha * A_shared_prev + (1.0 - alpha) * A_shared_candidate
                A_shared = A_shared / A_shared.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            E_dt = self.temp_res_scorer(xt, self.geom_bias)
            E_df = self.freq_res_scorer(xf, self.geom_bias)

            Delta_t = torch.tanh(E_dt) * self.residual_scale
            Delta_f = torch.tanh(E_df) * self.residual_scale

            A_t = self._normalize_adj(torch.log(A_shared.clamp_min(1e-8)) + Delta_t)
            A_f = self._normalize_adj(torch.log(A_shared.clamp_min(1e-8)) + Delta_f)

            shared_list.append(A_shared)
            temp_list.append(A_t)
            freq_list.append(A_f)
            delta_t_list.append(Delta_t)
            delta_f_list.append(Delta_f)

            A_shared_prev = A_shared

        A_shared_seq = torch.stack(shared_list, dim=1)
        A_t_seq = torch.stack(temp_list, dim=1)
        A_f_seq = torch.stack(freq_list, dim=1)
        Delta_t_seq = torch.stack(delta_t_list, dim=1)
        Delta_f_seq = torch.stack(delta_f_list, dim=1)

        return A_shared_seq, A_t_seq, A_f_seq, Delta_t_seq, Delta_f_seq


# ============================================================
# Graph encoder
# ============================================================

class DynamicGraphBlockFromAdj(nn.Module):
    """
    Applies GCN over provided dynamic adjacency sequence.
    Input:
        x:     [B, W, C, D]
        adjs:  [B, W, C, C]
    Output:
        out:   [B, W, C, D]
    """
    def __init__(self, d_model, num_nodes, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes

        self.conv = GCNConv(d_model, d_model, add_self_loops=False, normalize=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjs):
        B, W, C, D = x.shape
        outs = []

        for w in range(W):
            xw = x[:, w, :, :]      # [B, C, D]
            Aw = adjs[:, w, :, :]   # [B, C, C]

            edge_index_b, edge_weight_b = build_batched_edge_index_from_adjs(Aw)
            xw_flat = xw.reshape(B * C, D)
            residual = xw_flat

            out = self.conv(xw_flat, edge_index_b, edge_weight_b)
            out = F.gelu(out)
            out = self.dropout(out)
            out = self.norm(out)
            out = out + residual
            out = out.view(B, C, D)
            outs.append(out)

        return torch.stack(outs, dim=1)  # [B, W, C, D]


# ============================================================
# Fusion / pooling / heads
# ============================================================

class NodeLevelDualFusion(nn.Module):
    """
    Fuse temporal and frequency graph outputs at node level.

    Input:
        x_t, x_f: [B, W, C, D]
    Output:
        x: [B, W, C, D]
    """
    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        hidden = max(32, d_model)
        self.gate = nn.Sequential(
            nn.Linear(4 * d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Sigmoid()
        )
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_t, x_f):
        h = torch.cat([x_t, x_f, x_t - x_f, x_t * x_f], dim=-1)
        g = self.gate(h)
        x = g * x_t + (1.0 - g) * x_f
        return self.out_proj(x)


class ChannelAttentionFusion(nn.Module):
    """
    Input : [B, W, C, D]
    Output: [B, W, D]
    """
    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        hidden = max(16, d_model // 2)
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
        self.value = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        s = self.score(x)              # [B, W, C, 1]
        w = torch.softmax(s, dim=2)
        v = self.value(x)
        g = self.gate(x)
        return (w * g * v).sum(dim=2)  # [B, W, D]


class SequenceAttentionPooling(nn.Module):
    """
    Input : [B, W, D]
    Output: [B, D]
    """
    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        hidden = max(16, d_model // 2)
        self.pool = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        w = torch.softmax(self.pool(x), dim=1)
        return (x * w).sum(dim=1)


class BranchHead(nn.Module):
    """
    Auxiliary branch classifier from [B, W, C, D].
    """
    def __init__(self, d_model, num_classes=2, dropout=0.3):
        super().__init__()
        self.channel_fusion = ChannelAttentionFusion(d_model, dropout=dropout)
        self.seq_pool = SequenceAttentionPooling(d_model, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        x = self.channel_fusion(x)   # [B, W, D]
        x = self.seq_pool(x)         # [B, D]
        logits = self.classifier(x)  # [B, num_classes]
        return logits, x


# ============================================================
# Main model
# ============================================================

class EEGEvoDualGraphGRU(nn.Module):
    """
    Novel version:
      1) normalize raw EEG
      2) build temporal and frequency node features
      3) learn temporally evolving shared-specific graphs
      4) run graph encoders using branch-specific graphs
      5) auxiliary heads
      6) node-level dual fusion
      7) channel fusion
      8) GRU
      9) attention pooling
      10) final classifier
    """
    def __init__(
        self,
        input_dim=19,
        seq_len=100,
        num_classes=2,
        sfreq=100,
        num_windows=5,
        temporal_branch_dim=4,
        node_dim=64,
        graph_hidden_dim=64,
        gru_hidden=64,
        gru_layers=1,
        dropout=0.3,
        bidirectional=True,
        graph_topk=8,
        graph_temperature=0.7,
        graph_geom_mix=0.2,
        residual_scale=0.5,
        freq_bands=((4, 8), (8, 13), (13, 30), (30, 45)),
        local_window_size=20,
        aux_loss_weight=0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.sfreq = sfreq
        self.num_windows = num_windows
        self.node_dim = node_dim
        self.aux_loss_weight = aux_loss_weight

        self.channel_names = CGX_CHANNEL_NAMES[:input_dim]
        geom_bias = build_geometry_bias(self.channel_names, sigma=2.0, self_loop=True)
        self.register_buffer("geom_bias", geom_bias)

        # -------- feature builders --------
        self.temporal_stem = MultiScaleTemporalStem(
            input_dim=input_dim,
            branch_dim=temporal_branch_dim,
            kernels=(3, 7),
            dropout=dropout,
        )

        self.band_encoder = DynamicLocalBandPowerEncoder(
            sfreq=sfreq,
            bands=freq_bands,
            window_size=local_window_size,
            use_relative=True,
            log_power=True,
        )

        temporal_in_dim = 1 + self.temporal_stem.out_dim
        freq_in_dim = len(freq_bands) * 2

        self.temporal_proj = nn.Sequential(
            nn.Linear(temporal_in_dim, node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.frequency_proj = nn.Sequential(
            nn.Linear(freq_in_dim, node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.temporal_window = WindowAggregator(seq_len=seq_len, num_windows=num_windows)
        self.frequency_window = WindowAggregator(seq_len=seq_len, num_windows=num_windows)

        # -------- evolving shared-specific graph learner --------
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

        # -------- graph branches --------
        self.temporal_graph = DynamicGraphBlockFromAdj(
            d_model=node_dim,
            num_nodes=input_dim,
            dropout=dropout,
        )
        self.frequency_graph = DynamicGraphBlockFromAdj(
            d_model=node_dim,
            num_nodes=input_dim,
            dropout=dropout,
        )

        # -------- auxiliary heads --------
        self.temporal_aux_head = BranchHead(node_dim, num_classes=num_classes, dropout=dropout)
        self.frequency_aux_head = BranchHead(node_dim, num_classes=num_classes, dropout=dropout)

        # -------- fusion + sequence modeling --------
        self.dual_fusion = NodeLevelDualFusion(node_dim, dropout=dropout)
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

        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, gru_out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, num_classes),
        )

    def _normalize(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-5)
        return (x - mean) / std

    def _build_temporal_nodes(self, x):
        raw = x.unsqueeze(-1)                  # [B, T, C, 1]
        temp = self.temporal_stem(x)           # [B, T, C, D_t]
        x = torch.cat([raw, temp], dim=-1)     # [B, T, C, 1+D_t]
        x = self.temporal_proj(x)              # [B, T, C, D]
        x = self.temporal_window(x)            # [B, W, C, D]
        return x

    def _build_frequency_nodes(self, x):
        freq = self.band_encoder(x)            # [B, T, C, F]
        freq = self.frequency_proj(freq)       # [B, T, C, D]
        freq = self.frequency_window(freq)     # [B, W, C, D]
        return freq

    def forward(self, x, return_features=False):
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, C], got {x.shape}")
        if x.size(1) != self.seq_len or x.size(2) != self.input_dim:
            raise ValueError(f"Expected [B, {self.seq_len}, {self.input_dim}], got {x.shape}")

        # 1) normalize
        x_norm = self._normalize(x)

        # 2) branch node features
        x_t_nodes = self._build_temporal_nodes(x_norm)   # [B, W, C, D]
        x_f_nodes = self._build_frequency_nodes(x_norm)  # [B, W, C, D]

        # 3) evolving shared-specific graph learning
        A_shared, A_t, A_f, Delta_t, Delta_f = self.graph_learner(x_t_nodes, x_f_nodes)

        # 4) graph encoders
        x_t_graph = self.temporal_graph(x_t_nodes, A_t)
        x_f_graph = self.frequency_graph(x_f_nodes, A_f)

        # 5) auxiliary heads
        temp_logits, temp_embed = self.temporal_aux_head(x_t_graph)
        freq_logits, freq_embed = self.frequency_aux_head(x_f_graph)

        # 6) node-level fusion
        x_dual = self.dual_fusion(x_t_graph, x_f_graph)  # [B, W, C, D]

        # 7) collapse channels
        x_seq = self.channel_fusion(x_dual)              # [B, W, D]
        x_seq = self.seq_dropout(x_seq)

        # 8) GRU
        x_seq, _ = self.gru(x_seq)                       # [B, W, H]
        x_seq = self.post_gru_dropout(x_seq)

        # 9) attention pooling
        x_global = self.seq_pool(x_seq)                  # [B, H]

        # 10) final classifier
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
                "temporal_graph": x_t_graph,
                "frequency_graph": x_f_graph,
                "dual_graph": x_dual,
                "temp_embed": temp_embed,
                "freq_embed": freq_embed,
                "seq_features": x_seq,
                "global_features": x_global,
            })

        return out


# ============================================================
# Loss
# ============================================================

def graph_regularization_loss(
    outputs,
    geom_bias,
    smooth_weight=0.10,
    shared_weight=0.10,
    residual_weight=0.02,
    geom_weight=0.05,
):
    """
    Graph-aware regularization terms.

    outputs must include:
        A_shared: [B, W, C, C]
        A_temp:   [B, W, C, C]
        A_freq:   [B, W, C, C]
        Delta_temp, Delta_freq: [B, W, C, C]
    """
    A_shared = outputs["A_shared"]
    A_temp = outputs["A_temp"]
    A_freq = outputs["A_freq"]
    Delta_temp = outputs["Delta_temp"]
    Delta_freq = outputs["Delta_freq"]

    loss_smooth = 0.0
    if A_shared.size(1) > 1:
        loss_smooth = ((A_shared[:, 1:] - A_shared[:, :-1]) ** 2).mean()

    loss_shared = ((A_temp - A_shared) ** 2).mean() + ((A_freq - A_shared) ** 2).mean()

    # Encourage branch residuals to stay informative but controlled
    loss_residual = (Delta_temp ** 2).mean() + (Delta_freq ** 2).mean()

    # Penalize long-range / geometry-inconsistent edges
    # geom_bias high = locally plausible; penalize (1 - geom)
    geom_penalty = (1.0 - geom_bias).unsqueeze(0).unsqueeze(0)  # [1,1,C,C]
    loss_geom = (geom_penalty * (A_shared + A_temp + A_freq) / 3.0).mean()

    total_graph_reg = (
        smooth_weight * loss_smooth +
        shared_weight * loss_shared +
        residual_weight * loss_residual +
        geom_weight * loss_geom
    )

    stats = {
        "loss_graph_smooth": float(loss_smooth.detach().item()) if torch.is_tensor(loss_smooth) else float(loss_smooth),
        "loss_graph_shared": float(loss_shared.detach().item()),
        "loss_graph_residual": float(loss_residual.detach().item()),
        "loss_graph_geom": float(loss_geom.detach().item()),
    }

    return total_graph_reg, stats


def evo_dual_graph_loss(
    model,
    outputs,
    targets,
    label_smoothing=0.05,
    aux_weight=0.3,
    smooth_weight=0.10,
    shared_weight=0.10,
    residual_weight=0.02,
    geom_weight=0.05,
):
    """
    Total loss:
        L = L_fused
            + aux_weight * L_temp
            + aux_weight * L_freq
            + L_graph_reg
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    loss_fused = criterion(outputs["fused_logits"], targets)
    loss_temp = criterion(outputs["temp_logits"], targets)
    loss_freq = criterion(outputs["freq_logits"], targets)

    loss_graph_reg, graph_stats = graph_regularization_loss(
        outputs=outputs,
        geom_bias=model.geom_bias,
        smooth_weight=smooth_weight,
        shared_weight=shared_weight,
        residual_weight=residual_weight,
        geom_weight=geom_weight,
    )

    loss = loss_fused + aux_weight * loss_temp + aux_weight * loss_freq + loss_graph_reg

    stats = {
        "loss_total": loss.item(),
        "loss_fused": loss_fused.item(),
        "loss_temp": loss_temp.item(),
        "loss_freq": loss_freq.item(),
        "loss_graph_reg": loss_graph_reg.item(),
    }
    stats.update(graph_stats)

    return loss, stats


def build_optimizer(model, lr=1e-3, weight_decay=5e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


# ============================================================
# Sanity test
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = EEGEvoDualGraphGRU(
        input_dim=19,
        seq_len=100,
        num_classes=2,
        sfreq=100,
        num_windows=10,
        temporal_branch_dim=8,
        node_dim=64,
        graph_hidden_dim=64,
        gru_hidden=64,
        gru_layers=3,
        dropout=0.2,
        bidirectional=True,
        graph_topk=8,
        graph_temperature=0.7,
        graph_geom_mix=0.2,
        residual_scale=0.5,
        freq_bands=((4, 8), (8, 13), (13, 30), (30, 45)),
        local_window_size=20,
        aux_loss_weight=0.3,
    ).to(device)

    x = torch.randn(8, 100, 19).to(device)
    y = torch.randint(0, 2, (8,), device=device)

    outputs = model(x, return_features=True)
    loss, stats = evo_dual_graph_loss(
        model=model,
        outputs=outputs,
        targets=y,
        label_smoothing=0.05,
        aux_weight=0.3,
        smooth_weight=0.10,
        shared_weight=0.10,
        residual_weight=0.02,
        geom_weight=0.05,
    )
    loss.backward()

    size_mb, num_params = calculate_model_size(model)

    print("=" * 90)
    print("Sanity check")
    print("Input shape              :", tuple(x.shape))
    print("Temporal nodes           :", tuple(outputs["temporal_nodes"].shape))
    print("Frequency nodes          :", tuple(outputs["frequency_nodes"].shape))
    print("A_shared                 :", tuple(outputs["A_shared"].shape))
    print("A_temp                   :", tuple(outputs["A_temp"].shape))
    print("A_freq                   :", tuple(outputs["A_freq"].shape))
    print("Temporal graph           :", tuple(outputs["temporal_graph"].shape))
    print("Frequency graph          :", tuple(outputs["frequency_graph"].shape))
    print("Dual graph               :", tuple(outputs["dual_graph"].shape))
    print("Sequence features        :", tuple(outputs["seq_features"].shape))
    print("Global features          :", tuple(outputs["global_features"].shape))
    print("Fused logits             :", tuple(outputs["fused_logits"].shape))
    print(f"Loss total               : {stats['loss_total']:.4f}")
    print(f"Loss fused               : {stats['loss_fused']:.4f}")
    print(f"Loss temp                : {stats['loss_temp']:.4f}")
    print(f"Loss freq                : {stats['loss_freq']:.4f}")
    print(f"Loss graph reg           : {stats['loss_graph_reg']:.4f}")
    print(f"Model size               : {size_mb:.2f} MB")
    print(f"Parameters               : {num_params}")