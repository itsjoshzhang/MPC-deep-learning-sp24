import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split




class ModelTrainer:
    def __init__(self, model: NoiseModel, comment='', **optim_params):
        self.model = model

        self.loss = nn.KLDivLoss(reduction='mean')
        self.optimizer = Adam(**optim_params)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min')
        self.writer = SummaryWriter(comment=comment)

    def training_step(self, train_loader, val_loader, epoch):
        self.model.train()
        train_losses = []
        for data, label in train_loader:
            pred = self.model(data)
            loss = self.loss(pred, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss.item())
        mean_train_loss = np.mean(train_losses)
        self.writer.add_scalar('train_loss', mean_train_loss, global_step=epoch)
        self.scheduler.step(np.mean(train_losses))

        self.model.eval()
        val_losses = []
        with torch.no_grad():
            for data, label in val_loader:
                pred = self.model(data)
                loss = self.loss(pred, label)
                val_losses.append(loss.item())
        mean_val_loss = np.mean(val_losses)
        self.writer.add_scalar('val_loss', mean_val_loss, global_step=epoch)
        return

    def fit(self, X, y, n_epochs, batch_size, **params):
        """Warning: At this point, y should already be x_{k+1} - f_nominal(x_k, u_k). """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True,
                                                            random_state=params['seed'])
        train_loader = DataLoader(X_train, y_train, batch_size=batch_size)
        val_loader = DataLoader(X_test, y_test, batch_size=batch_size)
        for epoch in range(n_epochs):
            self.training_step(train_loader, val_loader, epoch)





if __name__ == '__main__':
    parse_data()
    exit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

    params = vars(parser.parse_args())
    np.random.seed(params['seed'])

    # Dataset: a sequence of x_k, u_k.
    # Dataset transform:
    # 1. Use the nominal model, get x_{k + 1} - f_nominal(x_k, u_k).
    # 2.1 With history of length h, use a sliding window to concatenate h consecutive x and u to form a single input.
    # 2.2 With a sequential model, pass in the entire sequence as a single dataset. (?)

    nominal_model = CasadiDynamicCLBicycle(...)
    noise_model = NoiseModel(...)
    model = CasadiDynamicCLBicycleNoise(nominal_model, noise_model)
    trainer = ModelTrainer()
