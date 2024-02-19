import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class Robot_RNN(nn.Module):

    def __init__(self, inputs, hidden, output, layers):
        super(Robot_RNN, self).__init__()

        self.lstm = nn.LSTM(inputs, hidden, layers, batch_first=True)
        self.full = nn.Linear(hidden, output)

    def forward(self, x):
        # x: [batch_size, seql, inputs]
        lstm_out, (hn, cn) = self.lstm(x)

        # Select the output of the last time step
        output = lstm_out[:, -1, :]
        output = self.full(output)
        return output

class RobotData(Dataset):

    def __init__(self, data, seql):
        self.data = data
        self.seql = seql

    def __len__(self):
        return len(self.data) - self.seql

    def __getitem__(self, index):
        seql  = self.data[index: index + self.seql]
        label = self.data[index + self.seql][:6]    # Assuming the next state is the label

        return (torch.tensor(seql, dtype=torch.float),
                torch.tensor(label, dtype=torch.float))

def create_sequences(data, seql):
    sequences = []
    labels = []

    for i in range(len(data) - seql):
        sequence = data[i: i + seql]  # Grab seql number of messages
        label = data[i + seql]        # The next message is the label

        sequences.append(sequence)
        labels.append(label)
    return sequences, labels

def normalize_data(data):
    tens = torch.tensor(data, dtype=torch.float32)
    mean = torch.mean(tens, dim=0)

    std  = torch.std (tens, dim=0)
    norm = (tens - mean) / std
    return norm, mean, std

# Hyperparameters
INPUTS = 8      # Assuming 8 features (excluding <time> and including accelerations)
HIDDEN = 128    # Can be tuned
OUTPUT = 6      # Predicting positions, orientation, and velocities
LAYERS = 8      # Can be tuned

LEARN_R = 0.001
BATCH_S = 64
EPOCH_N = 128

def whatthefuck():

    # Assuming 'data' is your preprocessed dataset
    DATA = pickle.load(open("data_new.pkl", "rb"))

    dataset = RobotData(DATA, 10)
    train_loader = DataLoader(dataset, batch_size=BATCH_S, shuffle=True)

    all_sequences = []
    all_labels = []

    for test in DATA:
        norm_test,_,_ = normalize_data(test)  # Normalize each test
        sequences, labels = create_sequences(norm_test, seql=10)  # Create sequences

        all_sequences.extend(sequences)
        all_labels.extend(labels)

    # Convert to PyTorch tensors or a DataLoader for training
    sequences = torch.stack(all_sequences)
    labels = torch.stack(all_labels)

    # Model, Loss, and Optimizer
    model = Robot_RNN(INPUTS, HIDDEN, OUTPUT, LAYERS)
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
    optimizer = optim.Adam(model.parameters(), lr=LEARN_R)

    # Training Loop
    for epoch in range(EPOCH_N):
        for i, (sequences, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCH_N}], Loss: {loss.item():.4f}')