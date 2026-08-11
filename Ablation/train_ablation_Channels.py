import os
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

from dataloader_5folds_ablation import get_dataloader, CGX_CHANNEL_NAMES
from EEGEvoDualGraphGRU_ablation import EEGEvoDualGraphGRUAblation, evo_dual_graph_loss, calculate_model_size


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_normal_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.zeros_(param.data)


def get_logits_from_output(outputs):
    if isinstance(outputs, dict):
        return outputs["fused_logits"]
    return outputs


def compute_loss(model, outputs, labels, criterion, label_smoothing=0.05, aux_weight=0.3, smooth_weight=0.10, shared_weight=0.10, residual_weight=0.02, geom_weight=0.05):
    if isinstance(outputs, dict):
        return evo_dual_graph_loss(
            model=model,
            outputs=outputs,
            targets=labels,
            label_smoothing=label_smoothing,
            aux_weight=aux_weight,
            smooth_weight=smooth_weight,
            shared_weight=shared_weight,
            residual_weight=residual_weight,
            geom_weight=geom_weight,
        )
    loss = criterion(outputs, labels)
    return loss, None


def train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip=1.0, label_smoothing=0.05, aux_weight=0.3, smooth_weight=0.10, shared_weight=0.10, residual_weight=0.02, geom_weight=0.05):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    running_stats = {
        "loss_total": 0.0, "loss_fused": 0.0, "loss_temp": 0.0, "loss_freq": 0.0,
        "loss_graph_reg": 0.0, "loss_graph_smooth": 0.0, "loss_graph_shared": 0.0,
        "loss_graph_residual": 0.0, "loss_graph_geom": 0.0,
    }
    num_stat_batches = 0

    for batch_data, batch_labels in train_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(batch_data)
        logits = get_logits_from_output(outputs)
        loss, loss_stats = compute_loss(model, outputs, batch_labels, criterion, label_smoothing, aux_weight, smooth_weight, shared_weight, residual_weight, geom_weight)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        running_loss += loss.item() * batch_data.size(0)
        preds = torch.argmax(logits, dim=1)
        correct_predictions += (preds == batch_labels).sum().item()
        total_samples += batch_labels.size(0)
        if loss_stats is not None:
            for k in running_stats:
                if k in loss_stats:
                    running_stats[k] += loss_stats[k]
            num_stat_batches += 1

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct_predictions / total_samples
    mean_stats = {k: v / max(1, num_stat_batches) for k, v in running_stats.items()}
    return epoch_loss, epoch_accuracy, mean_stats


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    all_labels, all_preds = [], []
    for batch_data, batch_labels in data_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device).long()
        outputs = model(batch_data)
        logits = get_logits_from_output(outputs)
        preds = torch.argmax(logits, dim=1)
        all_labels.extend(batch_labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return accuracy, f1, all_preds, all_labels


def create_model(device, feature_dim, seq_len, num_classes, config):
    model = EEGEvoDualGraphGRUAblation(
        input_dim=feature_dim,
        seq_len=seq_len,
        num_classes=num_classes,
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
        channel_names=config["channel_names"],
        branch_mode=config["branch_mode"],
    ).to(device)
    initialize_weights(model)
    return model


def run_experiment(exp_name, config, base_dir, mat_file_path, car_drone_ind, device):
    feature_dim = len(config["channel_names"])
    seq_len = 100
    num_classes = 2
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 1e-4
    num_epochs = 60
    num_repeats = 1
    grad_clip = 1.0
    label_smoothing = 0.05
    aux_weight = 0.30
    smooth_weight = 0.10
    shared_weight = 0.10
    residual_weight = 0.02
    geom_weight = 0.05

    output_dir = Path(base_dir) / f"ablation_{exp_name}"
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    results_file = log_dir / "cross_validation_results.txt"
    save_json_path = log_dir / "summary.json"

    temp_model = create_model(device, feature_dim, seq_len, num_classes, config)
    size_mb, num_params = calculate_model_size(temp_model)
    del temp_model

    all_repeat_fold_accs, all_repeat_fold_f1s = [], []
    fold_records = []

    with open(results_file, "w") as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(json.dumps(config, indent=2) + "\n")
        f.write(f"Model size: {size_mb:.2f} MB | Params: {num_params}\n")
        f.write("=" * 80 + "\n")

    for repeat_idx in range(num_repeats):
        repeat_fold_accs, repeat_fold_f1s = [], []
        for fold in range(5):
            train_loader = get_dataloader(
                file_path=mat_file_path,
                split="train",
                fold=fold,
                car_drone_ind=car_drone_ind,
                batch_size=batch_size,
                shuffle=True,
                channel_names=config["channel_names"],
            )
            test_loader = get_dataloader(
                file_path=mat_file_path,
                split="test",
                fold=fold,
                car_drone_ind=car_drone_ind,
                batch_size=batch_size,
                shuffle=False,
                channel_names=config["channel_names"],
            )

            model = create_model(device, feature_dim, seq_len, num_classes, config)
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

            best_test_accuracy, best_f1_score, best_epoch, best_state_dict = 0.0, 0.0, 0, None
            for epoch in range(num_epochs):
                train_loss, train_acc, train_stats = train_one_epoch(
                    model=model,
                    train_loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    grad_clip=grad_clip,
                    label_smoothing=label_smoothing,
                    aux_weight=aux_weight,
                    smooth_weight=smooth_weight,
                    shared_weight=shared_weight,
                    residual_weight=residual_weight,
                    geom_weight=geom_weight,
                )
                test_accuracy, f1, _, _ = evaluate(model, test_loader, device)
                scheduler.step()
                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    best_f1_score = f1
                    best_epoch = epoch + 1
                    best_state_dict = copy.deepcopy(model.state_dict())
                print(
                    f"[{exp_name}] Fold [{fold + 1}/5] Epoch [{epoch + 1}/{num_epochs}] | "
                    f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Test Acc: {test_accuracy:.4f} | Test F1: {f1:.4f}"
                )

            if best_state_dict is not None:
                model.load_state_dict(best_state_dict)
            ckpt_path = ckpt_dir / f"repeat_{repeat_idx:02d}_fold_{fold:02d}_best.pt"
            torch.save(
                {
                    "experiment": exp_name,
                    "config": config,
                    "repeat_idx": repeat_idx,
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "best_test_accuracy": best_test_accuracy,
                    "best_f1_score": best_f1_score,
                    "model_state_dict": model.state_dict(),
                },
                ckpt_path,
            )
            repeat_fold_accs.append(best_test_accuracy)
            repeat_fold_f1s.append(best_f1_score)
            fold_records.append({
                "repeat": repeat_idx,
                "fold": fold,
                "best_epoch": best_epoch,
                "best_accuracy": best_test_accuracy,
                "best_f1": best_f1_score,
            })
            with open(results_file, "a") as f:
                f.write(
                    f"Repeat {repeat_idx + 1}, Fold {fold + 1}: epoch={best_epoch}, "
                    f"acc={best_test_accuracy:.4f}, f1={best_f1_score:.4f}\n"
                )
            del model
            torch.cuda.empty_cache()

        all_repeat_fold_accs.extend(repeat_fold_accs)
        all_repeat_fold_f1s.extend(repeat_fold_f1s)

    summary = {
        "experiment": exp_name,
        "config": config,
        "model_size_mb": size_mb,
        "num_params": num_params,
        "overall_accuracy_mean": float(np.mean(all_repeat_fold_accs)),
        "overall_accuracy_std": float(np.std(all_repeat_fold_accs)),
        "overall_f1_mean": float(np.mean(all_repeat_fold_f1s)),
        "overall_f1_std": float(np.std(all_repeat_fold_f1s)),
        "fold_records": fold_records,
    }
    with open(results_file, "a") as f:
        f.write("=" * 80 + "\n")
        f.write(json.dumps(summary, indent=2) + "\n")
    with open(save_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    # set_seed(42)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mat_file_path = os.path.join(base_dir, "Data", "spilit_full_loop.mat")
    car_drone_ind = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiments = {
        "without_fp1_fp2": {
            "channel_names": [ch for ch in CGX_CHANNEL_NAMES if ch not in {"Fp1", "Fp2"}],
            "branch_mode": "dual",
        },
        "only_fp1_fp2": {
            "channel_names": ["Fp1", "Fp2"],
            "branch_mode": "dual",
        },
        "temporal_only_19ch": {
            "channel_names": list(CGX_CHANNEL_NAMES),
            "branch_mode": "temporal_only",
        },
        "frequency_only_19ch": {
            "channel_names": list(CGX_CHANNEL_NAMES),
            "branch_mode": "frequency_only",
        },
    }

    all_summaries = {}
    for exp_name, config in experiments.items():
        print("\n" + "#" * 100)
        print(f"Running experiment: {exp_name}")
        print(json.dumps(config, indent=2))
        print("#" * 100)
        all_summaries[exp_name] = run_experiment(exp_name, config, base_dir, mat_file_path, car_drone_ind, device)

    compare_path = Path(base_dir) / "ablation_comparison_summary.json"
    with open(compare_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\nFinished all ablations.")
    print(f"Saved comparison summary to: {compare_path}")


if __name__ == "__main__":
    main()
