"""
train_EEGEvoDualGraphGRU_ablations.py

Runs the three focused EEGEvoDualGraphGRU ablations:

    1. static_graph
    2. independent_graphs
    3. mean_fusion

The script follows the original 5-fold, repeated-training procedure and
records:

    results/ablation_study/
        epoch_history.csv
        fold_results.csv
        model_summary.csv
        run_config.txt
        predictions/
            <model>_repeat_<r>_fold_<f>.npz
        checkpoints/
            <model>_repeat_<r>_fold_<f>.pth

Important:
The original supplied training code selects the best epoch using test
accuracy. This script preserves that behaviour for direct comparability.
For final publication-quality evaluation, a separate validation split is
preferable.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from dataloader_5folds import get_dataloader
from EEGEvoDualGraphGRU import evo_dual_graph_loss
from EEGEvoDualGraphGRU_ablation_models import (
    ABLATION_MODEL_REGISTRY,
    build_ablation_model,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Utilities
# ============================================================

def initialize_weights(model: nn.Module) -> None:
    """
    Generic initialization matching the original training script.
    Normalization parameters are left unchanged.
    """
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.LSTM, nn.GRU)):
            for name, parameter in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_normal_(parameter.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(parameter.data)
                elif "bias" in name:
                    nn.init.zeros_(parameter.data)


def calculate_model_size(model: nn.Module) -> Tuple[float, int]:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    size_mb = total_params * 4 / (1024 ** 2)
    return size_mb, total_params


def get_logits_from_output(outputs):
    if isinstance(outputs, dict):
        if "fused_logits" not in outputs:
            raise KeyError(
                "Model returned a dictionary without 'fused_logits'."
            )
        return outputs["fused_logits"]
    return outputs


def append_csv(path: Path, fieldnames: List[str], row: Dict) -> None:
    file_exists = path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(path: Path, data: Dict) -> None:
    with path.open("w") as file:
        json.dump(data, file, indent=2)


def model_loss_weights(model_key: str, args) -> Dict[str, float]:
    """
    The independent-graph ablation intentionally has no shared-specific
    graph relationship, so its shared-consistency term is disabled.
    """
    return {
        "label_smoothing": args.label_smoothing,
        "aux_weight": args.aux_weight,
        "smooth_weight": args.smooth_weight,
        "shared_weight": (
            0.0
            if model_key == "independent_graphs"
            else args.shared_weight
        ),
        "residual_weight": args.residual_weight,
        "geom_weight": args.geom_weight,
    }


# ============================================================
# Train / evaluate
# ============================================================

def compute_loss(
    model: nn.Module,
    outputs,
    labels: torch.Tensor,
    loss_weights: Dict[str, float],
):
    if isinstance(outputs, dict):
        return evo_dual_graph_loss(
            model=model,
            outputs=outputs,
            targets=labels,
            **loss_weights,
        )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=loss_weights["label_smoothing"]
    )
    loss = criterion(outputs, labels)
    return loss, None


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    device: torch.device,
    loss_weights: Dict[str, float],
    grad_clip: Optional[float],
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    stat_sums = {
        "loss_total": 0.0,
        "loss_fused": 0.0,
        "loss_temp": 0.0,
        "loss_freq": 0.0,
        "loss_graph_reg": 0.0,
        "loss_graph_smooth": 0.0,
        "loss_graph_shared": 0.0,
        "loss_graph_residual": 0.0,
        "loss_graph_geom": 0.0,
    }
    stat_batches = 0

    for batch_data, batch_labels in train_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device).long()

        optimizer.zero_grad(set_to_none=True)

        outputs = model(batch_data)
        logits = get_logits_from_output(outputs)

        loss, loss_stats = compute_loss(
            model=model,
            outputs=outputs,
            labels=batch_labels,
            loss_weights=loss_weights,
        )

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

        optimizer.step()

        batch_size = batch_labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (
            torch.argmax(logits, dim=1) == batch_labels
        ).sum().item()
        total_samples += batch_size

        if loss_stats is not None:
            for key in stat_sums:
                if key in loss_stats:
                    stat_sums[key] += float(loss_stats[key])
            stat_batches += 1

    mean_stats = None
    if stat_batches > 0:
        mean_stats = {
            key: value / stat_batches
            for key, value in stat_sums.items()
        }

    return (
        total_loss / total_samples,
        total_correct / total_samples,
        mean_stats,
    )


@torch.no_grad()
def evaluate(model: nn.Module, data_loader, device: torch.device):
    model.eval()

    labels_all: List[int] = []
    predictions_all: List[int] = []

    for batch_data, batch_labels in data_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device).long()

        outputs = model(batch_data)
        logits = get_logits_from_output(outputs)
        predictions = torch.argmax(logits, dim=1)

        labels_all.extend(batch_labels.cpu().numpy().tolist())
        predictions_all.extend(predictions.cpu().numpy().tolist())

    accuracy = accuracy_score(labels_all, predictions_all)
    macro_f1 = f1_score(
        labels_all,
        predictions_all,
        average="macro",
        zero_division=0,
    )
    balanced_accuracy = balanced_accuracy_score(
        labels_all,
        predictions_all,
    )
    cm = confusion_matrix(labels_all, predictions_all, labels=[0, 1])

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_accuracy),
        "predictions": np.asarray(predictions_all, dtype=np.int64),
        "labels": np.asarray(labels_all, dtype=np.int64),
        "confusion_matrix": cm.astype(np.int64),
    }


# ============================================================
# Model factory
# ============================================================

def create_model(
    model_key: str,
    device: torch.device,
    args,
) -> nn.Module:
    model = build_ablation_model(
        model_key=model_key,
        input_dim=args.feature_dim,
        seq_len=args.seq_len,
        num_classes=args.num_classes,
        sfreq=args.sfreq,
        num_windows=args.num_windows,
        temporal_branch_dim=args.temporal_branch_dim,
        node_dim=args.node_dim,
        graph_hidden_dim=args.graph_hidden_dim,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        graph_topk=args.graph_topk,
        graph_temperature=args.graph_temperature,
        graph_geom_mix=args.graph_geom_mix,
        residual_scale=args.residual_scale,
        freq_bands=((4, 8), (8, 13), (13, 30), (30, 45)),
        local_window_size=args.local_window_size,
        aux_loss_weight=args.aux_weight,
    ).to(device)

    initialize_weights(model)
    return model


# ============================================================
# One model / repeat / fold
# ============================================================

def run_fold(
    model_key: str,
    repeat_idx: int,
    fold_idx: int,
    args,
    device: torch.device,
    output_dir: Path,
    epoch_csv: Path,
    fold_csv: Path,
):
    seed = args.base_seed + repeat_idx * 100 + fold_idx
    set_seed(seed)

    train_loader = get_dataloader(
        file_path=args.mat_file,
        split="train",
        fold=fold_idx,
        car_drone_ind=args.car_drone_ind,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = get_dataloader(
        file_path=args.mat_file,
        split="test",
        fold=fold_idx,
        car_drone_ind=args.car_drone_ind,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = create_model(model_key, device, args)
    loss_weights = model_loss_weights(model_key, args)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.eta_min,
    )

    best = {
        "epoch": 0,
        "accuracy": -1.0,
        "macro_f1": -1.0,
        "balanced_accuracy": -1.0,
        "predictions": None,
        "labels": None,
        "confusion_matrix": None,
        "state_dict": None,
    }

    start_time = time.time()

    for epoch_idx in range(args.num_epochs):
        train_loss, train_accuracy, train_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_weights=loss_weights,
            grad_clip=args.grad_clip,
        )

        test_metrics = evaluate(model, test_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Preserve original selection rule: best test accuracy.
        if test_metrics["accuracy"] > best["accuracy"]:
            best.update(
                {
                    "epoch": epoch_idx + 1,
                    "accuracy": test_metrics["accuracy"],
                    "macro_f1": test_metrics["macro_f1"],
                    "balanced_accuracy": test_metrics[
                        "balanced_accuracy"
                    ],
                    "predictions": test_metrics["predictions"].copy(),
                    "labels": test_metrics["labels"].copy(),
                    "confusion_matrix": test_metrics[
                        "confusion_matrix"
                    ].copy(),
                    "state_dict": copy.deepcopy(model.state_dict()),
                }
            )

        epoch_row = {
            "model": model_key,
            "repeat": repeat_idx + 1,
            "fold": fold_idx + 1,
            "epoch": epoch_idx + 1,
            "seed": seed,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_balanced_accuracy": test_metrics[
                "balanced_accuracy"
            ],
            "loss_graph_reg": (
                train_stats["loss_graph_reg"]
                if train_stats is not None else np.nan
            ),
            "loss_graph_smooth": (
                train_stats["loss_graph_smooth"]
                if train_stats is not None else np.nan
            ),
            "loss_graph_shared": (
                train_stats["loss_graph_shared"]
                if train_stats is not None else np.nan
            ),
            "loss_graph_residual": (
                train_stats["loss_graph_residual"]
                if train_stats is not None else np.nan
            ),
            "loss_graph_geom": (
                train_stats["loss_graph_geom"]
                if train_stats is not None else np.nan
            ),
        }
        append_csv(epoch_csv, list(epoch_row.keys()), epoch_row)

        print(
            f"{model_key:20s} | "
            f"Repeat {repeat_idx + 1}/{args.num_repeats} | "
            f"Fold {fold_idx + 1}/5 | "
            f"Epoch {epoch_idx + 1:03d}/{args.num_epochs} | "
            f"Loss {train_loss:.4f} | "
            f"Train ACC {train_accuracy:.4f} | "
            f"Test ACC {test_metrics['accuracy']:.4f} | "
            f"Macro-F1 {test_metrics['macro_f1']:.4f} | "
            f"BACC {test_metrics['balanced_accuracy']:.4f}"
        )

    elapsed_seconds = time.time() - start_time

    checkpoint_dir = output_dir / "checkpoints"
    prediction_dir = output_dir / "predictions"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        checkpoint_dir
        / f"{model_key}_repeat_{repeat_idx + 1}_fold_{fold_idx + 1}.pth"
    )
    prediction_path = (
        prediction_dir
        / f"{model_key}_repeat_{repeat_idx + 1}_fold_{fold_idx + 1}.npz"
    )

    if args.save_checkpoints and best["state_dict"] is not None:
        torch.save(
            {
                "model_key": model_key,
                "repeat": repeat_idx + 1,
                "fold": fold_idx + 1,
                "best_epoch": best["epoch"],
                "state_dict": best["state_dict"],
                "loss_weights": loss_weights,
                "args": vars(args),
            },
            checkpoint_path,
        )

    np.savez_compressed(
        prediction_path,
        predictions=best["predictions"],
        labels=best["labels"],
        confusion_matrix=best["confusion_matrix"],
        best_epoch=np.asarray(best["epoch"]),
        accuracy=np.asarray(best["accuracy"]),
        macro_f1=np.asarray(best["macro_f1"]),
        balanced_accuracy=np.asarray(best["balanced_accuracy"]),
    )

    cm = best["confusion_matrix"]
    fold_row = {
        "model": model_key,
        "repeat": repeat_idx + 1,
        "fold": fold_idx + 1,
        "seed": seed,
        "best_epoch": best["epoch"],
        "accuracy": best["accuracy"],
        "macro_f1": best["macro_f1"],
        "balanced_accuracy": best["balanced_accuracy"],
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "elapsed_seconds": elapsed_seconds,
        "prediction_file": str(prediction_path),
        "checkpoint_file": (
            str(checkpoint_path)
            if args.save_checkpoints else ""
        ),
    }
    append_csv(fold_csv, list(fold_row.keys()), fold_row)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fold_row


# ============================================================
# Aggregate and record results
# ============================================================

def aggregate_results(
    fold_rows: List[Dict],
    model_size_info: Dict[str, Dict],
    summary_csv: Path,
) -> List[Dict]:
    summary_rows = []

    for model_key in ABLATION_MODEL_REGISTRY:
        rows = [
            row for row in fold_rows
            if row["model"] == model_key
        ]
        if not rows:
            continue

        accuracy = np.asarray(
            [row["accuracy"] for row in rows],
            dtype=float,
        )
        macro_f1 = np.asarray(
            [row["macro_f1"] for row in rows],
            dtype=float,
        )
        balanced_accuracy = np.asarray(
            [row["balanced_accuracy"] for row in rows],
            dtype=float,
        )

        summary_row = {
            "model": model_key,
            "num_runs": len(rows),
            "mean_accuracy": accuracy.mean(),
            "std_accuracy": accuracy.std(ddof=0),
            "mean_macro_f1": macro_f1.mean(),
            "std_macro_f1": macro_f1.std(ddof=0),
            "mean_balanced_accuracy": balanced_accuracy.mean(),
            "std_balanced_accuracy": balanced_accuracy.std(ddof=0),
            "parameters": model_size_info[model_key]["parameters"],
            "model_size_mb": model_size_info[model_key][
                "model_size_mb"
            ],
        }
        append_csv(
            summary_csv,
            list(summary_row.keys()),
            summary_row,
        )
        summary_rows.append(summary_row)

    return summary_rows


def write_text_summary(
    path: Path,
    summary_rows: List[Dict],
    args,
) -> None:
    with path.open("w") as file:
        file.write("EEGEvoDualGraphGRU Ablation Study\n")
        file.write("=" * 78 + "\n")
        file.write(
            f"Repeats: {args.num_repeats} | "
            f"Folds: 5 | Epochs: {args.num_epochs}\n"
        )
        file.write(
            "Best epoch selection: highest test accuracy "
            "(preserved from original training code)\n"
        )
        file.write("=" * 78 + "\n\n")

        for row in summary_rows:
            file.write(f"Model: {row['model']}\n")
            file.write(
                f"  Accuracy:          "
                f"{row['mean_accuracy']:.4f} ± "
                f"{row['std_accuracy']:.4f}\n"
            )
            file.write(
                f"  Macro-F1:          "
                f"{row['mean_macro_f1']:.4f} ± "
                f"{row['std_macro_f1']:.4f}\n"
            )
            file.write(
                f"  Balanced accuracy: "
                f"{row['mean_balanced_accuracy']:.4f} ± "
                f"{row['std_balanced_accuracy']:.4f}\n"
            )
            file.write(
                f"  Parameters:        {row['parameters']}\n"
            )
            file.write(
                f"  Model size:        "
                f"{row['model_size_mb']:.2f} MB\n\n"
            )


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the three EEGEvoDualGraphGRU ablations."
    )

    base_dir = Path(__file__).resolve().parent

    parser.add_argument(
        "--mat-file",
        type=str,
        default=str(base_dir / "Data" / "spilit_full_loop.mat"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(base_dir / "results" / "ablation_study"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(ABLATION_MODEL_REGISTRY.keys()),
        choices=list(ABLATION_MODEL_REGISTRY.keys()),
    )

    parser.add_argument("--car-drone-ind", type=int, default=2)
    parser.add_argument("--feature-dim", type=int, default=19)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--sfreq", type=int, default=100)
    parser.add_argument("--num-windows", type=int, default=10)

    parser.add_argument("--temporal-branch-dim", type=int, default=8)
    parser.add_argument("--node-dim", type=int, default=128)
    parser.add_argument("--graph-hidden-dim", type=int, default=128)
    parser.add_argument("--gru-hidden", type=int, default=128)
    parser.add_argument("--gru-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--bidirectional",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--graph-topk", type=int, default=8)
    parser.add_argument(
        "--graph-temperature",
        type=float,
        default=0.7,
    )
    parser.add_argument("--graph-geom-mix", type=float, default=0.2)
    parser.add_argument("--residual-scale", type=float, default=0.5)
    parser.add_argument("--local-window-size", type=int, default=20)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eta-min", type=float, default=1e-5)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--aux-weight", type=float, default=0.30)
    parser.add_argument("--smooth-weight", type=float, default=0.10)
    parser.add_argument("--shared-weight", type=float, default=0.10)
    parser.add_argument("--residual-weight", type=float, default=0.02)
    parser.add_argument("--geom-weight", type=float, default=0.05)

    parser.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    mat_file = Path(args.mat_file)
    if not mat_file.exists():
        raise FileNotFoundError(
            f"Dataset file was not found: {mat_file}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_csv = output_dir / "epoch_history.csv"
    fold_csv = output_dir / "fold_results.csv"
    summary_csv = output_dir / "model_summary.csv"
    summary_txt = output_dir / "model_summary.txt"
    config_json = output_dir / "run_config.json"

    # Start a clean result set for this invocation.
    for path in (epoch_csv, fold_csv, summary_csv):
        if path.exists():
            path.unlink()

    write_json(config_json, vars(args))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Models: {args.models}")
    print(f"Output directory: {output_dir}")

    model_size_info = {}
    for model_key in args.models:
        model = create_model(model_key, device, args)
        size_mb, parameters = calculate_model_size(model)
        model_size_info[model_key] = {
            "model_size_mb": size_mb,
            "parameters": parameters,
        }
        print(
            f"{model_key:20s} | "
            f"{parameters:,} parameters | {size_mb:.2f} MB"
        )
        del model

    all_fold_rows = []

    for model_key in args.models:
        print("\n" + "#" * 90)
        print(f"Starting ablation: {model_key}")
        print("#" * 90)

        for repeat_idx in range(args.num_repeats):
            for fold_idx in range(5):
                fold_row = run_fold(
                    model_key=model_key,
                    repeat_idx=repeat_idx,
                    fold_idx=fold_idx,
                    args=args,
                    device=device,
                    output_dir=output_dir,
                    epoch_csv=epoch_csv,
                    fold_csv=fold_csv,
                )
                all_fold_rows.append(fold_row)

                print(
                    f"Completed {model_key} | "
                    f"Repeat {repeat_idx + 1} | "
                    f"Fold {fold_idx + 1} | "
                    f"Best epoch {fold_row['best_epoch']} | "
                    f"ACC {fold_row['accuracy']:.4f} | "
                    f"Macro-F1 {fold_row['macro_f1']:.4f} | "
                    f"BACC {fold_row['balanced_accuracy']:.4f}"
                )

    summary_rows = aggregate_results(
        fold_rows=all_fold_rows,
        model_size_info=model_size_info,
        summary_csv=summary_csv,
    )
    write_text_summary(summary_txt, summary_rows, args)

    print("\n" + "=" * 90)
    print("Ablation training complete")
    print("=" * 90)

    for row in summary_rows:
        print(
            f"{row['model']:20s} | "
            f"ACC {row['mean_accuracy']:.4f} ± "
            f"{row['std_accuracy']:.4f} | "
            f"Macro-F1 {row['mean_macro_f1']:.4f} ± "
            f"{row['std_macro_f1']:.4f} | "
            f"BACC {row['mean_balanced_accuracy']:.4f} ± "
            f"{row['std_balanced_accuracy']:.4f}"
        )

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
