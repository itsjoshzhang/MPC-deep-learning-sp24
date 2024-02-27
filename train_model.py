import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RobotData(Dataset):

    def __init__(self, data, size):
        featrs, labels = [], []
        self.size = size

        for bag in data:
            for i in range(len(bag)-size-1):    # -1 to ensure <size> messages + 1 for label
                featr = bag[i: i + size]        # Extract sequence of length <size> messages

                featrs.append(featr)
                label = bag[i + size][:6]       # Next state (excluding time / acceleration)
                labels.append(label)

        self.featrs = torch.tensor(featrs, dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels, dtype=torch.float32).to(device)
        self.norm()

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

    def __init__(self, inputs, hidden, output, layers, state_f):
        super(GRU_Model, self).__init__()

        self.hidden = hidden
        self.layers = layers
        self.state_f = state_f
        self.h_state = None

        self.gru = nn.GRU(inputs, hidden, layers, batch_first=True)
        self.full_c = nn.Linear(hidden, output)

    def init_hidden(self, batch_s):
        self.h_state = torch.zeros(self.layers, batch_s, self.hidden).to(device)

    def forward(self, x):
        if not self.state_f or self.h_state is None:
            self.init_hidden(x.size(0))

        # Detach to avoid backprop thru time
        output, self.h_state = self.gru(x, self.h_state.detach())
        return self.full_c(output[:, -1, :]) # shape (BATCH_S, OUTPUT)

class LSTM_Model(nn.Module):

    def __init__(self, inputs, hidden, output, layers, state_f):
        super(LSTM_Model, self).__init__()

        self.hidden = hidden
        self.layers = layers
        self.state_f = state_f
        self.h_state = None

        self.lstm = nn.LSTM(inputs, hidden, layers, batch_first=True)
        self.full_c = nn.Linear(hidden, output)

    def init_hidden(self, batch_s):
        self.h_state = (torch.zeros(self.layers, batch_s, self.hidden).to(device),
                        torch.zeros(self.layers, batch_s, self.hidden).to(device))

    def forward(self, x):
        if not self.state_f or self.h_state is None:
            self.init_hidden(x.size(0))

        unhook = (self.h_state[0].detach(), self.h_state[1].detach())
        output, self.h_state = self.lstm(x, unhook)
        return self.full_c(output[:, -1, :]) # shape (BATCH_S, OUTPUT)

INPUTS = 8      # (pos.(2), orient.(1), vel.(3), acceleration (2))
OUTPUT = 6      # (positions (2), orientation (1), velocities (3))
HIDDEN = 16
LAYERS = 4

FT_SIZE = 4
BATCH_S = 16
LEARN_R = 0.001
STATE_F = False

DATA = pickle.load(open("data_new.pkl", "rb"))
dataset = RobotData(DATA, FT_SIZE)

def train_model(model_type):

    t_size = int(len(dataset) * 0.8)
    v_size = len(dataset) - t_size

    t_data, v_data = random_split(dataset, [t_size, v_size])
    t_load = DataLoader(t_data, batch_size=BATCH_S, shuffle=True)
    v_load = DataLoader(v_data, batch_size=BATCH_S, shuffle=False)

    model  = model_type(INPUTS, HIDDEN, OUTPUT, LAYERS, STATE_F).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARN_R)
    scheduler = ReduceLROnPlateau(optimizer)

    criterion = nn.L1Loss()
    min_loss = float('inf')
    patience = 0

    name = str(model_type).split(".")[1].split("_")[0]
    path = f"models/{name}_{HIDDEN}_{LAYERS}_{FT_SIZE}_{BATCH_S}_{STATE_F}.pt"
    print(f"Training {path}:")

    for i in range(100):
        model.train()
        for featrs, labels in t_load:
            featrs = featrs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(featrs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for featrs, labels in v_load:
                featrs = featrs.to(device)
                labels = labels.to(device)
            
                output = model(featrs)
                loss = criterion(output, labels)
                v_loss += loss.item()
        
        v_loss /= len(v_load)
        scheduler.step(v_loss)
        print(f"Epoch {i + 1}: valid_loss = {v_loss:.4f}")

        if v_loss < min_loss:
            min_loss = v_loss
            patience = 0
            torch.save(model.state_dict(), path)
        else: patience += 1

        if patience > 10:
            print("Early stop triggered.")
        if v_loss > 0.2:
            print("Bad loss. Restarting:")
            return 1

if __name__ == "__main__":
    for hidden in [16, 64]:
        for layers in [4, 8]:
            for batch_s in [16, 64]:
                for state_f in [True, False]:

                    HIDDEN, LAYERS, BATCH_S, STATE_F = hidden, layers, batch_s, state_f
                    while train_model(GRU_Model):  continue
                    while train_model(LSTM_Model): continue