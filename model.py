import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class RobotData(Dataset):

    def __init__(self, data, size):
        self.featrs = []
        self.labels = []
        self.size = size

        for bag in data:
            for i in range(len(bag)-size-1):    # -1 to ensure <size> messages + 1 for label
                featr = bag[i: i + size]        # Extract sequence of length <size> messages

                self.featrs.append(featr)
                label = bag[i+size+1][:6]       # Next state (excluding time / accelerations)
                self.labels.append(label)

    def __len__(self):
        return len(self.featrs)

    def __getitem__(self, i):
        tens = lambda x: torch.tensor(x, dtype=torch.float64)
        return tens(self.featrs[i]), tens(self.labels[i])

# Assuming `data` is your dataset loaded in the required format
dataset = RobotData(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
