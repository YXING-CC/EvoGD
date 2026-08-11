"""
EEGEvoDualGraphGRU_ablation_models.py

Three focused ablation variants for EEGEvoDualGraphGRU:

1. static_graph
   Replaces temporally evolving graphs with one sample-specific
   shared-specific graph repeated across all windows.

2. independent_graphs
   Removes shared-specific decomposition and learns temporal and
   frequency graphs independently for each window.

3. mean_fusion
   Replaces learnable node-level gated dual fusion with an
   equal elementwise mean.

Place this file in the same directory as EEGEvoDualGraphGRU.py.
"""

from __future__ import annotations

from typing import Dict, Type

import torch
import torch.nn as nn

from EEGEvoDualGraphGRU import (
    EEGEvoDualGraphGRU,
    EvolvingSharedSpecificGraphLearner,
)


# ============================================================
# Ablation 1: static shared-specific graphs
# ============================================================

class StaticSharedSpecificGraphLearner(EvolvingSharedSpecificGraphLearner):
    """
    Learn one sample-specific shared-specific graph from the
    window-averaged temporal and frequency node features, then repeat
    that graph across all windows.

    This removes temporal graph evolution while preserving:
      - sample-specific graph learning;
      - shared graph construction;
      - temporal/frequency residual graphs;
      - geometry prior;
      - graph sparsification and normalization.
    """

    def forward(self, x_t: torch.Tensor, x_f: torch.Tensor):
        # x_t, x_f: [B, W, C, D]
        _, num_windows, _, _ = x_t.shape

        xt = x_t.mean(dim=1)  # [B, C, D]
        xf = x_f.mean(dim=1)  # [B, C, D]

        geom = self.geom_bias.unsqueeze(0)  # [1, C, C]

        e_t_base = self.temp_base_scorer(xt, self.geom_bias)
        e_f_base = self.freq_base_scorer(xf, self.geom_bias)

        e_shared = 0.5 * (e_t_base + e_f_base)
        e_shared = (
            (1.0 - self.geom_mix) * e_shared
            + self.geom_mix * geom
        )
        a_shared = self._normalize_adj(e_shared)

        delta_t = (
            torch.tanh(self.temp_res_scorer(xt, self.geom_bias))
            * self.residual_scale
        )
        delta_f = (
            torch.tanh(self.freq_res_scorer(xf, self.geom_bias))
            * self.residual_scale
        )

        a_t = self._normalize_adj(
            torch.log(a_shared.clamp_min(1e-8)) + delta_t
        )
        a_f = self._normalize_adj(
            torch.log(a_shared.clamp_min(1e-8)) + delta_f
        )

        def repeat_windows(tensor: torch.Tensor) -> torch.Tensor:
            return (
                tensor.unsqueeze(1)
                .expand(-1, num_windows, -1, -1)
                .contiguous()
            )

        return (
            repeat_windows(a_shared),
            repeat_windows(a_t),
            repeat_windows(a_f),
            repeat_windows(delta_t),
            repeat_windows(delta_f),
        )


# ============================================================
# Ablation 2: independent temporal/frequency graphs
# ============================================================

class IndependentBranchGraphLearner(EvolvingSharedSpecificGraphLearner):
    """
    Learn temporal and frequency graphs independently for each window.

    Neither branch graph is generated from a common shared graph.
    A_shared is returned only for interface compatibility and is the
    normalized arithmetic mean of A_temp and A_freq.

    For this ablation, set shared_weight=0.0 during training because
    shared-specific consistency is intentionally removed.
    """

    def forward(self, x_t: torch.Tensor, x_f: torch.Tensor):
        # x_t, x_f: [B, W, C, D]
        _, num_windows, _, _ = x_t.shape

        shared_list = []
        temp_list = []
        freq_list = []
        delta_t_list = []
        delta_f_list = []

        geom = self.geom_bias.unsqueeze(0)

        for window_idx in range(num_windows):
            xt = x_t[:, window_idx]
            xf = x_f[:, window_idx]

            e_t = self.temp_base_scorer(xt, self.geom_bias)
            e_f = self.freq_base_scorer(xf, self.geom_bias)

            e_t = (
                (1.0 - self.geom_mix) * e_t
                + self.geom_mix * geom
            )
            e_f = (
                (1.0 - self.geom_mix) * e_f
                + self.geom_mix * geom
            )

            a_t = self._normalize_adj(e_t)
            a_f = self._normalize_adj(e_f)

            # Compatibility tensor only; it does not generate A_t or A_f.
            a_shared = 0.5 * (a_t + a_f)
            a_shared = a_shared / a_shared.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)

            delta_t = a_t - a_shared
            delta_f = a_f - a_shared

            shared_list.append(a_shared)
            temp_list.append(a_t)
            freq_list.append(a_f)
            delta_t_list.append(delta_t)
            delta_f_list.append(delta_f)

        return (
            torch.stack(shared_list, dim=1),
            torch.stack(temp_list, dim=1),
            torch.stack(freq_list, dim=1),
            torch.stack(delta_t_list, dim=1),
            torch.stack(delta_f_list, dim=1),
        )


# ============================================================
# Ablation 3: mean node-level fusion
# ============================================================

class MeanNodeFusion(nn.Module):
    """
    Parameter-free equal fusion:
        X_fused = 0.5 * (X_temporal + X_frequency)
    """

    def forward(
        self,
        x_t: torch.Tensor,
        x_f: torch.Tensor,
    ) -> torch.Tensor:
        return 0.5 * (x_t + x_f)


# ============================================================
# Complete model variants
# ============================================================

class EEGEvoDualGraphGRUStaticGraph(EEGEvoDualGraphGRU):
    """Full architecture with static sample-specific shared-specific graphs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.graph_learner = StaticSharedSpecificGraphLearner(
            d_model=self.node_dim,
            num_nodes=self.input_dim,
            geom_bias=self.geom_bias,
            hidden_dim=kwargs.get("graph_hidden_dim", 64),
            topk=kwargs.get("graph_topk", 8),
            temperature=kwargs.get("graph_temperature", 0.7),
            geom_mix=kwargs.get("graph_geom_mix", 0.2),
            residual_scale=kwargs.get("residual_scale", 0.5),
        )


class EEGEvoDualGraphGRUIndependentGraphs(EEGEvoDualGraphGRU):
    """Full architecture without shared-specific graph decomposition."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.graph_learner = IndependentBranchGraphLearner(
            d_model=self.node_dim,
            num_nodes=self.input_dim,
            geom_bias=self.geom_bias,
            hidden_dim=kwargs.get("graph_hidden_dim", 64),
            topk=kwargs.get("graph_topk", 8),
            temperature=kwargs.get("graph_temperature", 0.7),
            geom_mix=kwargs.get("graph_geom_mix", 0.2),
            residual_scale=kwargs.get("residual_scale", 0.5),
        )


class EEGEvoDualGraphGRUMeanFusion(EEGEvoDualGraphGRU):
    """Full architecture with parameter-free mean node fusion."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dual_fusion = MeanNodeFusion()


ABLATION_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "static_graph": EEGEvoDualGraphGRUStaticGraph,
    "independent_graphs": EEGEvoDualGraphGRUIndependentGraphs,
    "mean_fusion": EEGEvoDualGraphGRUMeanFusion,
}


def build_ablation_model(model_key: str, **model_kwargs) -> nn.Module:
    """
    Construct one of the three ablation models.

    Valid keys:
        static_graph
        independent_graphs
        mean_fusion
    """
    if model_key not in ABLATION_MODEL_REGISTRY:
        valid = ", ".join(ABLATION_MODEL_REGISTRY)
        raise ValueError(
            f"Unknown ablation model '{model_key}'. Valid keys: {valid}"
        )

    return ABLATION_MODEL_REGISTRY[model_key](**model_kwargs)


if __name__ == "__main__":
    common_kwargs = dict(
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
    )

    sample = torch.randn(2, 100, 19)

    for key in ABLATION_MODEL_REGISTRY:
        model = build_ablation_model(key, **common_kwargs)
        model.eval()

        with torch.no_grad():
            outputs = model(sample)

        print(
            f"{key:20s} | "
            f"logits={tuple(outputs['fused_logits'].shape)} | "
            f"A_shared={tuple(outputs['A_shared'].shape)}"
        )
