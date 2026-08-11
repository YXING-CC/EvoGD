# EvoDG Ablation Studies

This folder contains the ablation studies used to evaluate the contribution of different EEG inputs and key components of EvoDG.

## 1. Channel and Representation Ablation

Files:
EEGEvoDualGraphGRU_ablation_Channels.py
train_ablation_Channels.py

Four configurations are evaluated:

* **without_fp1_fp2** – removes Fp1 and Fp2 while retaining the other 17 EEG channels.
* **only_fp1_fp2** – uses only Fp1 and Fp2.
* **temporal_only_19ch** – uses all 19 channels with only the temporal branch.
* **frequency_only_19ch** – uses all 19 channels with only the frequency branch.

These experiments evaluate the contribution of frontal EEG channels and the complementary roles of temporal and frequency EEG representations.

Run:
python train_ablation_Channels.py

## 2. Model Module Ablation

Files:
EEGEvoDualGraphGRU_ablation_models.py
train_ablations_Models.py

Three key components of EvoDG are evaluated:

* **static_graph** – replaces the temporally evolving graph with a static graph to evaluate the contribution of dynamic graph evolution.
* **independent_graphs** – learns temporal and frequency graphs independently to evaluate the contribution of shared-specific graph learning.
* **mean_fusion** – replaces learnable dual-view fusion with simple averaging to evaluate the contribution of adaptive feature fusion.

Run all module ablations:

python train_ablations_Models.py


The experiments follow the same five-fold evaluation protocol as the full EvoDG model. Accuracy and Macro-F1 are used as the main performance metrics.

Please refer to the main project README and `Data/README.txt` for model, dataset, and environment details.
