import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split

class RobotData(Dataset):

    def __init__(self, data, size, norm):
        featrs, labels = [], []
        self.size = size

        for bag in data:
            for i in range(len(bag)-size-1):    # -1 to ensure <size> messages + 1 for label
                featr = bag[i: i + size]        # Extract sequence of length <size> messages

                self.featrs.append(featr)
                label = bag[i + size][:6]       # Next state (excluding time / acceleration)
                self.labels.append(label)

        tens = lambda x: torch.tensor(x, dtype = torch.float64)
        self.featrs = tens(featrs)
        self.labels = tens(labels)
        if norm: self.norm()

    def norm(self):
        f_mean = torch.mean(self.featrs, dim=0)
        f_std  = torch.std (self.featrs, dim=0)
        l_mean = torch.mean(self.labels, dim=0)
        l_std  = torch.std (self.labels, dim=0)

        self.featrs = (self.featrs - f_mean) / f_std
        self.labels = (self.labels - l_mean) / l_std

    def __len__(self):
        return len(self.featrs)

    def __getitem__(self, i):
        return self.featrs[i], self.labels[i]


class GRU_Model(nn.Module):

    def __init__(self, inputs, hidden, output, layers, drop_r):
        super(GRU_Model, self).__init__()

        self.hidden = hidden
        self.layers = layers
        self.gru    = nn.GRU(inputs, hidden, layers,
            batch_first=True, dropout=drop_r)
        self.full_c = nn.Linear(hidden, output)

    def forward(self, x):
        # Init hidden state w/0           <BATCH_S>                  <CPU/GPU>
        hidden = torch.zeros(self.layers, x.size(0), self.hidden).to(x.device)
        # Forward propagate GRU
        output, _ = self.gru(x, hidden) # x/output: (BATCH_S, SEQ_LEN, IN/HID)
        # Decode last time step
        return self.full_c(output[:, -1, :]) # output shape: (BATCH_S, OUTPUT)


INPUTS = 8      # (pos.(2), orient.(1), vel.(3), acceleration (2))
HIDDEN = 128
OUTPUT = 6      # (positions (2), orientation (1), velocities (3))
LAYERS = 8
DROP_R = 0.1

F_SIZE  = 10
DO_NORM = False
BATCH_S = 128
LEARN_R = 0.001
EPOCHS  = 128

DATA = pickle.load(open("data_new.pkl", "rb"))
dataset = RobotData(DATA, F_SIZE, DO_NORM)

def train_model():
    t_size = int(len(dataset) * 0.8)
    v_size = len(dataset) - t_size

    t_data, v_data = random_split(dataset, [t_size, v_size])

    t_load = DataLoader(t_data, batch_size=BATCH_S, shuffle=True)
    v_load = DataLoader(v_data, batch_size=BATCH_S, shuffle=False)

    criterion = nn.MSELoss()
    model = GRU_Model(INPUTS, HIDDEN, OUTPUT, LAYERS, DROP_R)

    optimizer = optim.Adam(model.parameters(), lr=LEARN_R)
    scheduler = ReduceLROnPlateau(optimizer)

    min_loss = float('inf')
    patience = 0

    for epoch in range(EPOCHS):
        model.train()

        for featrs, labels in t_load:
            optimizer.zero_grad()
            output = model(featrs)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        model.eval()
        v_loss = 0

        with torch.no_grad():
            for featrs, labels in v_load:
                output = model(featrs)
                loss = criterion(output, labels)
                v_loss += loss.item()
        
        v_loss /= len(v_load)
        print(f"Epoch {epoch+1}: valid_loss = {v_loss:.4f}")

        if v_loss < min_loss:
            min_loss = v_loss
            patience = 0

        else: patience += 1
        if patience >= 16:
            print("Early stopping triggered.")
            break