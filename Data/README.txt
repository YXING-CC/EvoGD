# EvoDG Dataset

The processed EEG dataset used for training and evaluating EvoDG is not included directly in this GitHub repository because of its large file size.

## Download

The dataset can be downloaded from Google Drive:

https://drive.google.com/drive/folders/1RIXUxofO8Xk5Of8J--nOTO65-saZxwTE?usp=drive_link

After downloading, place the required `.mat` data file in this `Data/` directory.

The expected project structure is:

```text
EvoDG/
│
├── Data/
│   ├── README.txt
│   └── spilit_longer_loop.mat
│
├── dataloader_5folds.py
├── EEGEvoDualGraphGRU.py
└── train_eeg_evo_dual_graph.py
```

## Data Format

The provided dataset has been preprocessed and organised into predefined five-fold cross-validation partitions.

The MATLAB file contains the variable:

```text
Data_cell
```

which stores the training and testing EEG samples and their corresponding labels for each fold.

Each EEG trial used by EvoDG has the shape:

```text
100 × 19
```

corresponding to:

```text
100 time samples × 19 EEG channels
```

The 19 EEG channels are:

```text
F7, Fp1, Fp2, F8, F3, Fz, F4, C3, Cz,
P8, P7, Pz, P4, T3, P3, O1, O2, C4, T4
```

The data loader supports different experimental subsets through the `car_drone_ind` parameter:

```python
car_drone_ind = 1   # Ground-vehicle (car) trials
car_drone_ind = 2   # Aerial-vehicle (drone) trials
```

The exact indexing of the predefined training/test partitions is implemented in `dataloader_5folds.py`.

## Usage

Once the dataset has been downloaded and placed in the `Data/` directory, no additional data preparation is required for the provided training pipeline.

Run:

```bash
python train_eeg_evo_dual_graph.py
```

The training script will automatically load the predefined training and testing partitions for each of the five cross-validation folds.

## Custom Data

To apply EvoDG to another EEG dataset, organise each EEG sample in the format:

```text
[T, C]
```

where `T` is the number of temporal samples and `C` is the number of EEG channels.

The current EvoDG configuration expects:

```text
T = 100
C = 19
```

For datasets with a different storage format, modify `load_mat_file()` in `dataloader_5folds.py` accordingly.

## Data Usage

Please use the dataset for research purposes and cite the associated publication when using the data or EvoDG framework in academic work.

Citation information and additional dataset documentation will be updated following publication.
