import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader

# Step 1: Load the .mat file
def load_mat_file(file_path, split, fold, car_drone_ind):
    data = scipy.io.loadmat(file_path)
    # Assuming 'data' is the key for your 5x1 cell array
    # Adjust the key name based on your .mat file structure
    # For car, train and test inds are 0,2 - 3,5
    # For drone, train and test inds are 6,8 - 9,11

    if car_drone_ind == 1:
        print('using car data')
        train_ind = 0
        train_lab = 2
        test_ind = 3
        test_lab = 5
    elif car_drone_ind == 2:
        print('using drone data')
        train_ind = 6
        train_lab = 8
        test_ind = 9
        test_lab = 11
    else:
        print('using all data')
        # train_ind = 12                # binary classification
        # train_lab = 13
        # test_ind = 14
        # test_lab = 15

        train_ind = 16                  # 4 class classfication
        train_lab = 18
        test_ind = 17
        test_lab = 19

        # train_ind = 20                  # car2drone
        # train_lab = 21
        # test_ind = 9
        # test_lab = 11

        # train_ind = 22                  # drone2car
        # train_lab = 23
        # test_ind = 3
        # test_lab = 5

        # train_ind = 16                  # car vs drone
        # train_lab = 24
        # test_ind = 17
        # test_lab = 25


    mat_cells = data['Data_cell'][fold, 0]
    mat_cells = mat_cells[0, :]  # Flatten the single row

    if split == 'train':
        sequences = mat_cells[train_ind]  # 680x1 cell, each containing 100x19 arrays
        labels = mat_cells[train_lab].squeeze()  # 680x1 double, convert to 1D
    else:
        sequences = mat_cells[test_ind]  # 680x1 cell, each containing 100x19 arrays
        labels = mat_cells[test_lab].squeeze()  # 680x1 double, convert to 1D

    return sequences, labels

# Step 2: Create a PyTorch Dataset
class EEGDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.long)  # Convert to tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load sequence and label
        sequence = self.sequences[idx, 0]  # Extract the 100x19 array from the cell
        label = self.labels[idx]

        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)  # Convert to tensor
        # sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # Add batch dimension, this applies to the EmT model

        # Avoid re-wrapping tensors unnecessarily
        if isinstance(label, torch.Tensor):
            label_tensor = label.float()  # Ensure correct dtype
        else:
            label_tensor = torch.tensor(label, dtype=torch.float32)

        return sequence_tensor, label_tensor

# Step 3: DataLoader
def get_dataloader(file_path, split, fold,car_drone_ind, batch_size, shuffle=True):
    sequences, labels = load_mat_file(file_path, split, fold, car_drone_ind)
    dataset = EEGDataset(sequences, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Usage
# file_path = '.\Data\spilit_full_loop.mat'
# batch_size = 32
# split = 'test'
# fold = 0
# dataloader = get_dataloader(file_path, split, fold, batch_size)
#
# for batch_data, batch_labels in dataloader:
#     print(batch_data.size())  # Should print [batch_size, 500, 19]
#     print(batch_labels.size(), batch_labels[0])  # Should print [batch_size]
#     break

# Iterating through the DataLoader
# for batch_idx, (inputs, targets) in enumerate(dataloader):
#     print(f"Batch {batch_idx}:")
#     print(f"Inputs shape: {inputs.shape}")  # Should be [batch_size, 100, 19]
#     print(f"Targets shape: {targets.shape}")  # Should be [batch_size]

