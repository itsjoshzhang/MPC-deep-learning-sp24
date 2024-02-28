import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RobotData(Dataset):

    def __init__(self, data):
        featrs, labels = [], []

        for bag in data:
            for i in range(len(bag) - 1):
                featr = bag[i]

                featrs.append(featr)
                label = bag[i + 1][:6]
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

class Basic_Model(nn.Module):

    def __init__(self, inputs, hidden, output, layers, do_drop):
        super(Basic_Model, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(inputs, hidden))
        
        for _ in range(layers):
            self.layers.append(nn.Linear(hidden, hidden))
            if do_drop:
                self.layers.append(nn.Dropout(0.1))
        
        self.layers.append(nn.Linear(hidden, output))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x) # No activ func with reg. output

INPUTS = 8      # [pos.(2), orient.(1), vel.(3), acceleration (2)]
OUTPUT = 6      # [positions (2), orientation (1), velocities (3)]
HIDDEN = 16
LAYERS = 4

BATCH_S = 16
LEARN_R = 0.001
DO_DROP = False
EPOCHS  = 100

DATA = pickle.load(open("data_new.pkl", "rb"))
dataset = RobotData(DATA)

def train_model(model_type):
    """
    Returns None on successful training
    Returns 1 & exits when v_loss > 0.1
    Prints debug msg if v_loss plateaus
    """
    t_size = int(len(dataset) * 0.8)
    v_size = len(dataset) - t_size

    t_data, v_data = random_split(dataset, [t_size, v_size])
    t_load = DataLoader(t_data, batch_size=BATCH_S, shuffle=True)
    v_load = DataLoader(v_data, batch_size=BATCH_S, shuffle=False)

    model  = model_type(INPUTS, HIDDEN, OUTPUT, LAYERS, DO_DROP).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARN_R)
    scheduler = ReduceLROnPlateau(optimizer)

    criterion = nn.L1Loss()
    min_loss = float('inf')
    patience = 0

    name = str(model_type).split(".")[1].split("_")[0]
    path = f"m/{name}_{HIDDEN}_{LAYERS}_{BATCH_S}.pt" # File name must follow setup_n_eval
    print(f"Training {path}:")

    for i in range(EPOCHS):
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
        if v_loss > 0.1:
            print("Bad loss. Restarting:")
            return 1

if __name__ == "__main__":
    for hidden in [16, 64]:
        for layers in [1, 2, 3, 4]:
            for batch_s in [16, 32, 64]:
                HIDDEN, LAYERS, BATCH_S = hidden, layers, batch_s
                while train_model(Basic_Model): continue