import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, f1_score

from dataloader_5folds import get_dataloader
from EEGEvoDualGraphGRU import EEGEvoDualGraphGRU, evo_dual_graph_loss


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Utilities
# =========================================================

def initialize_weights(model):
    """
    Safe generic initialization.
    Avoids re-initializing normalization layers.
    """
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


def calculate_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    model_size_bytes = total_params * 4
    model_size_mb = model_size_bytes / (1024 ** 2)
    return model_size_mb, total_params


def get_logits_from_output(outputs):
    if isinstance(outputs, dict):
        if "fused_logits" in outputs:
            return outputs["fused_logits"]
        raise KeyError("Model returned dict, but 'fused_logits' was not found.")
    return outputs


def compute_loss(
    model,
    outputs,
    labels,
    criterion,
    label_smoothing=0.05,
    aux_weight=0.3,
    smooth_weight=0.10,
    shared_weight=0.10,
    residual_weight=0.02,
    geom_weight=0.05,
):
    """
    Supports:
      1) standard tensor output models
      2) EEGEvoDualGraphGRU-style dict output models
    """
    if isinstance(outputs, dict):
        loss, loss_stats = evo_dual_graph_loss(
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
        return loss, loss_stats
    else:
        loss = criterion(outputs, labels)
        return loss, None


# =========================================================
# Train / Eval
# =========================================================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    grad_clip=1.0,
    label_smoothing=0.05,
    aux_weight=0.3,
    smooth_weight=0.10,
    shared_weight=0.10,
    residual_weight=0.02,
    geom_weight=0.05,
):
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    running_stats = {
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
    num_stat_batches = 0

    for batch_data, batch_labels in train_loader:
        batch_data = batch_data.to(device)              # [B, 100, 19]
        batch_labels = batch_labels.to(device).long()   # [B]

        optimizer.zero_grad()

        outputs = model(batch_data)
        logits = get_logits_from_output(outputs)

        loss, loss_stats = compute_loss(
            model=model,
            outputs=outputs,
            labels=batch_labels,
            criterion=criterion,
            label_smoothing=label_smoothing,
            aux_weight=aux_weight,
            smooth_weight=smooth_weight,
            shared_weight=shared_weight,
            residual_weight=residual_weight,
            geom_weight=geom_weight,
        )

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

    mean_stats = None
    if num_stat_batches > 0:
        mean_stats = {k: v / num_stat_batches for k, v in running_stats.items()}

    return epoch_loss, epoch_accuracy, mean_stats


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    device,
):
    model.eval()

    all_labels = []
    all_preds = []

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


# =========================================================
# Model factory
# =========================================================

def create_model(device, feature_dim, seq_len, num_classes, dropout, model_key="evo_dual_graph_gru"):
    if model_key == "evo_dual_graph_gru":
        model = EEGEvoDualGraphGRU_Lite(
            input_dim=feature_dim,
            seq_len=seq_len,
            num_classes=num_classes,
            sfreq=100,
            num_windows=10,
            temporal_branch_dim=8,
            node_dim=128,
            # graph_hidden_dim=128,
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
        ).to(device)
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    initialize_weights(model)
    return model


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    # set_seed(42)

    # -----------------------------------------------------
    # Paths and data settings
    # -----------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    mat_file_path = os.path.join(BASE_DIR, "Data", "spilit_longer_loop.mat")

    car_drone_ind = 1   # 1 = car, 2 = drone

    # -----------------------------------------------------
    # Model / training config
    # -----------------------------------------------------
    feature_dim = 19
    seq_len = 100
    num_classes = 2
    dropout = 0.15

    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 1e-4
    num_epochs = 100
    num_repeats = 1

    grad_clip = 1.0

    # losses
    label_smoothing = 0.05
    aux_weight = 0.30
    smooth_weight = 0.10
    shared_weight = 0.10
    residual_weight = 0.02
    geom_weight = 0.05

    model_key = "evo_dual_graph_gru"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = f"{model_key}_"
    results_file = model_name + "cross_validation_results.txt"
    predictions_file = model_name + "pred_lab.txt"

    # -----------------------------------------------------
    # Init log files
    # -----------------------------------------------------
    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            f.write("5-Fold Cross-Validation Results\n")
            f.write("=" * 60 + "\n")

    if not os.path.exists(predictions_file):
        with open(predictions_file, "w") as f:
            f.write("Predictions and Ground Truths for 5-Fold Cross-Validation\n")
            f.write("=" * 60 + "\n")

    run_header = (
        "\n" + "#" * 80 + "\n"
        f"New Run | model_key={model_key} | repeats={num_repeats} | epochs={num_epochs} "
        f"| batch_size={batch_size} | lr={learning_rate} | weight_decay={weight_decay}\n"
        f"Loss weights | aux={aux_weight} | smooth={smooth_weight} | shared={shared_weight} "
        f"| residual={residual_weight} | geom={geom_weight}\n"
        + "#" * 80 + "\n"
    )

    with open(results_file, "a") as f:
        f.write(run_header)

    with open(predictions_file, "a") as f:
        f.write(run_header)

    # -----------------------------------------------------
    # Print model size once
    # -----------------------------------------------------
    temp_model = create_model(
        device=device,
        feature_dim=feature_dim,
        seq_len=seq_len,
        num_classes=num_classes,
        dropout=dropout,
        model_key=model_key,
    )
    size_mb, num_params = calculate_model_size(temp_model)
    print(f"Model key: {model_key}")
    print(f"Model size: {size_mb:.2f} MB")
    print(f"Number of parameters: {num_params}")
    del temp_model

    # -----------------------------------------------------
    # Cross-validation
    # -----------------------------------------------------
    all_repeat_fold_accs = []
    all_repeat_fold_f1s = []

    for repeat_idx in range(num_repeats):
        print("\n" + "=" * 80)
        print(f"Repeat {repeat_idx + 1}/{num_repeats}")
        print("=" * 80)

        repeat_fold_accs = []
        repeat_fold_f1s = []

        with open(results_file, "a") as f:
            f.write(f"\nRepeat {repeat_idx + 1}/{num_repeats}\n")
            f.write("=" * 60 + "\n")

        for fold in range(5):
            print("\n" + "-" * 80)
            print(f"Starting Fold {fold + 1}/5")
            print("-" * 80)

            train_loader = get_dataloader(
                file_path=mat_file_path,
                split="train",
                fold=fold,
                car_drone_ind=car_drone_ind,
                batch_size=batch_size,
                shuffle=True
            )

            test_loader = get_dataloader(
                file_path=mat_file_path,
                split="test",
                fold=fold,
                car_drone_ind=car_drone_ind,
                batch_size=batch_size,
                shuffle=False
            )

            model = create_model(
                device=device,
                feature_dim=feature_dim,
                seq_len=seq_len,
                num_classes=num_classes,
                dropout=dropout,
                model_key=model_key,
            )

            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                weight_decay=weight_decay,
            )

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=1e-5,
            )

            best_test_accuracy = 0.0
            best_f1_score = 0.0
            best_preds = []
            best_labels = []
            best_epoch = 0
            best_state_dict = None

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

                test_accuracy, f1, preds, labels = evaluate(
                    model=model,
                    data_loader=test_loader,
                    device=device,
                )

                scheduler.step()

                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    best_f1_score = f1
                    best_preds = preds
                    best_labels = labels
                    best_epoch = epoch + 1
                    best_state_dict = copy.deepcopy(model.state_dict())

                if train_stats is not None:
                    print(
                        f"Repeat [{repeat_idx + 1}/{num_repeats}] "
                        f"Fold [{fold + 1}/5] "
                        f"Epoch [{epoch + 1}/{num_epochs}] | "
                        f"Loss: {train_loss:.4f} | "
                        f"Train Acc: {train_acc:.4f} | "
                        f"Test Acc: {test_accuracy:.4f} | "
                        f"Test F1: {f1:.4f} | "
                        f"GraphReg: {train_stats['loss_graph_reg']:.4f} | "
                        f"Smooth: {train_stats['loss_graph_smooth']:.4f} | "
                        f"Shared: {train_stats['loss_graph_shared']:.4f} | "
                        f"Geom: {train_stats['loss_graph_geom']:.4f}"
                    )
                else:
                    print(
                        f"Repeat [{repeat_idx + 1}/{num_repeats}] "
                        f"Fold [{fold + 1}/5] "
                        f"Epoch [{epoch + 1}/{num_epochs}] | "
                        f"Loss: {train_loss:.4f} | "
                        f"Train Acc: {train_acc:.4f} | "
                        f"Test Acc: {test_accuracy:.4f} | "
                        f"Test F1: {f1:.4f}"
                    )

            if best_state_dict is not None:
                model.load_state_dict(best_state_dict)

            repeat_fold_accs.append(best_test_accuracy)
            repeat_fold_f1s.append(best_f1_score)

            with open(results_file, "a") as f:
                f.write(
                    f"Fold {fold + 1} best epoch / accuracy / F1: "
                    f"{best_epoch}, {best_test_accuracy:.4f}, {best_f1_score:.4f}\n"
                )
                f.write("-" * 60 + "\n")

            with open(predictions_file, "a") as f:
                f.write(f"Repeat {repeat_idx + 1}, Fold {fold + 1} Predictions (Best Accuracy):\n{best_preds}\n")
                f.write(f"Repeat {repeat_idx + 1}, Fold {fold + 1} Ground Truth (Best Accuracy):\n{best_labels}\n")
                f.write("-" * 60 + "\n")

            print(
                f"Fold {fold + 1} complete. "
                f"Best Epoch: {best_epoch}, "
                f"Best Test Accuracy: {best_test_accuracy:.4f}, "
                f"Best F1 Score: {best_f1_score:.4f}"
            )

            del model
            torch.cuda.empty_cache()

        repeat_mean_acc = float(np.mean(repeat_fold_accs))
        repeat_std_acc = float(np.std(repeat_fold_accs))
        repeat_mean_f1 = float(np.mean(repeat_fold_f1s))
        repeat_std_f1 = float(np.std(repeat_fold_f1s))

        all_repeat_fold_accs.extend(repeat_fold_accs)
        all_repeat_fold_f1s.extend(repeat_fold_f1s)

        with open(results_file, "a") as f:
            f.write(
                f"Repeat {repeat_idx + 1} mean accuracy: {repeat_mean_acc:.4f} ± {repeat_std_acc:.4f}\n"
            )
            f.write(
                f"Repeat {repeat_idx + 1} mean F1: {repeat_mean_f1:.4f} ± {repeat_std_f1:.4f}\n"
            )
            f.write("=" * 60 + "\n")

        print("\n" + "-" * 80)
        print(
            f"Repeat {repeat_idx + 1} summary | "
            f"Accuracy: {repeat_mean_acc:.4f} ± {repeat_std_acc:.4f} | "
            f"F1: {repeat_mean_f1:.4f} ± {repeat_std_f1:.4f}"
        )
        print("-" * 80)

    overall_mean_acc = float(np.mean(all_repeat_fold_accs))
    overall_std_acc = float(np.std(all_repeat_fold_accs))
    overall_mean_f1 = float(np.mean(all_repeat_fold_f1s))
    overall_std_f1 = float(np.std(all_repeat_fold_f1s))

    with open(results_file, "a") as f:
        f.write("\nOverall Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Overall mean accuracy: {overall_mean_acc:.4f} ± {overall_std_acc:.4f}\n")
        f.write(f"Overall mean F1: {overall_mean_f1:.4f} ± {overall_std_f1:.4f}\n")
        f.write("=" * 60 + "\n")

    print("\n" + "=" * 80)
    print("Training complete.")
    print(f"Overall Accuracy: {overall_mean_acc:.4f} ± {overall_std_acc:.4f}")
    print(f"Overall F1: {overall_mean_f1:.4f} ± {overall_std_f1:.4f}")
    print("=" * 80)