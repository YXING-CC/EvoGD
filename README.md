# EvoDG: Temporally Evolving Dual-View Graph Learning for EEG Classification

EvoDG is a deep learning framework for EEG-based cognitive state classification using **temporally evolving dual-view brain graphs**. The model jointly learns temporal and frequency representations of raw EEG signals and dynamically constructs shared and view-specific functional connectivity graphs across successive temporal windows.

The framework was developed for investigating EEG responses during human hazard perception in collaborative autonomous vehicle scenarios. The current implementation accepts raw EEG segments with a default shape of:

```text
[batch_size, 100, 19]
```

where each sample contains 100 temporal samples from 19 EEG channels.

## Overview

EvoDG consists of five main stages:

```text
Raw EEG [B, T, C]
       │
       ├─────────────────────┐
       │                     │
Temporal View          Frequency View
Multi-scale Conv       Local Band Power
       │                     │
       └──────────┬──────────┘
                  │
        Temporal Windowing
                  │
       ┌──────────┴──────────┐
       │                     │
 Temporal Graph        Frequency Graph
       │                     │
       └──── Shared–Specific ┘
          Graph Learning
                  │
        Dynamic Graph GCN
                  │
        Node-Level Fusion
                  │
        Channel Attention
                  │
             BiGRU
                  │
       Attention Pooling
                  │
           Classification
```

Rather than assuming a fixed EEG connectivity structure, EvoDG learns connectivity separately for successive temporal windows. A shared graph captures connectivity common to the temporal and frequency representations, while branch-specific residual graphs capture complementary information associated with each representation.

---

## Repository Structure

The core implementation contains three Python files:

```text
EvoDG/
│
├── dataloader_5folds.py
├── EEGEvoDualGraphGRU.py
├── train_eeg_evo_dual_graph.py
│
└── Data/
    └── spilit_longer_loop.mat
```

### `dataloader_5folds.py`

This file implements the EEG data loading pipeline for the predefined five-fold dataset.

Its main components are:

* loading MATLAB `.mat` EEG data;
* selecting training and testing data for each fold;
* selecting car, drone, or combined experimental conditions;
* converting EEG sequences to PyTorch tensors; and
* constructing PyTorch `DataLoader` objects.


### `EEGEvoDualGraphGRU.py`

This file contains the main **EvoDG architecture** and its associated loss functions.

The primary model class is:

```python
EEGEvoDualGraphGRU
```

## `train_eeg_evo_dual_graph.py`

This file provides the complete training and evaluation pipeline.

It performs:

* model initialisation;
* five-fold cross-validation;
* AdamW optimisation;
* cosine learning-rate scheduling;
* gradient clipping;
* multi-objective EvoDG loss calculation;
* Accuracy evaluation;
* Macro-F1 evaluation;
* best-epoch selection;
* prediction logging; and
* cross-fold performance aggregation.

The current default training configuration is approximately:

```python
feature_dim = 19
seq_len = 100
num_classes = 2

batch_size = 128
learning_rate = 1e-3
weight_decay = 1e-4
num_epochs = 100

grad_clip = 1.0
```

The optimiser is AdamW:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)
```

with cosine annealing:

```python
CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-5
)
```

---

## Requirements

The implementation requires Python and the following major packages:

```text
Python >= 3.9
PyTorch
PyTorch Geometric
NumPy
SciPy
scikit-learn
```

Example installation:

```bash
pip install numpy scipy scikit-learn
pip install torch
pip install torch-geometric
```

PyTorch and PyTorch Geometric should preferably be installed according to the CUDA version available on the target machine.

---


### Configure the experiment

Open:

```text
train_eeg_evo_dual_graph.py
```

and specify the required experimental condition.

For ground-vehicle EEG:

```python
car_drone_ind = 1
```

For aerial-vehicle EEG:

```python
car_drone_ind = 2
```

Training hyperparameters can also be changed directly in the configuration section.

### Train the model

Run:

```bash
python train_eeg_evo_dual_graph.py
```

The script automatically executes all five folds:

```text
Fold 1
Fold 2
Fold 3
Fold 4
Fold 5
```

For each epoch it reports training loss, training accuracy, test accuracy, Macro-F1, and the major graph regularisation terms.

### 4. Results

After training, the script generates two output files:

```text
evo_dual_graph_gru_cross_validation_results.txt
evo_dual_graph_gru_pred_lab.txt
```

The first records fold-level and overall classification performance.

The second stores predictions and corresponding ground-truth labels for subsequent statistical analysis.

The primary evaluation metrics are:

```text
Accuracy
Macro-F1
```

At the end of five-fold cross-validation, the mean and standard deviation across folds are reported.

---

## Using EvoDG with Another EEG Dataset

For a new dataset, EEG trials should first be converted to the following format:

```text
[B, T, C]
```

For example:

```python
x.shape
# [batch_size, 100, 19]
```

The model can then be instantiated as:

```python
from EEGEvoDualGraphGRU import EEGEvoDualGraphGRU

model = EEGEvoDualGraphGRU(
    input_dim=19,
    seq_len=100,
    num_classes=2,
    sfreq=100,
    num_windows=10,
    node_dim=64,
    gru_hidden=64,
    dropout=0.2
)
```

and called using:

```python
outputs = model(x)

logits = outputs["fused_logits"]
```

The model additionally exposes the learned connectivity structures:

```python
outputs["A_shared"]
outputs["A_temp"]
outputs["A_freq"]
outputs["Delta_temp"]
outputs["Delta_freq"]
```

These outputs can be used for subsequent analysis and visualisation of the learned EEG connectivity patterns.

For more detailed representations, use:

```python
outputs = model(x, return_features=True)
```

which additionally returns temporal/frequency node representations, graph embeddings, fused graph features, sequential features, and global EEG embeddings.

---

## Notes on the Current Code

Before running the repository, please ensure that the model class name used in the training script matches the class defined in the model file.

The provided model implementation defines:

```python
EEGEvoDualGraphGRU
```

Therefore, the model factory should instantiate:

```python
model = EEGEvoDualGraphGRU(...)
```

rather than:

```python
EEGEvoDualGraphGRU_Lite(...)
```

The filenames should also be standardised to:

```text
EEGEvoDualGraphGRU.py
train_eeg_evo_dual_graph.py
```

so that the imports in the training script work directly.

---


This implementation was developed for research into EEG-based human hazard awareness and human–autonomy collaboration involving heterogeneous ground and aerial autonomous systems.
