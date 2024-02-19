import torch
import torch.nn as nn
import torch.optim as optim

class Robot_RNN(nn.Module):
    def __init__(self, inputs, hidden, layers, output):

        super(Robot_RNN, self).__init__()
        self.hidden = hidden
        self.layers = layers
        
        self.lstm = nn.LSTM(inputs, hidden, layers, batch_first=True)
        self.full = nn.Linear(hidden, output)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layers, x.size(0), self.hidden).to(x.device)
        # Initialize cell state
        c0 = torch.zeros(self.layers, x.size(0), self.hidden).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.full(out[:, -1, :])
        return out