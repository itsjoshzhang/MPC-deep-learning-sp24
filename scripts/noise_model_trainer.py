import logging
import argparse
import numpy as np
import mpclab_common

import torch
import torch.nn as nn
import torch.optim as optim

from mpclab_common.models.dynamics_models import CasadiDynamicCLBicycle
from mpclab_common.models.model_types import DynamicBicycleConfig

from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from model_scripts.noise_model import FeedforwardNoiseModel, CasadiDynamicCLBicycleNoise

from utils.data_utils import NoiseDataset, DynamicsDataset, q_labels, u_labels
from utils.log import setup_custom_logger
from utils import pytorch_utils as ptu

class ModelTrainer:
    def __init__(self, model: FeedforwardNoiseModel, comment='', lr=1e-3, no_logging=False, behavior='noise'):
        self.model = model

        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.99)
        self.behavior = behavior

        self.writer = None
        self.logger = setup_custom_logger('noise_model_trainer', console_level=logging.INFO)
        if not no_logging:
            self.writer = SummaryWriter(comment=comment)

    def get_logger(self):
        return self.logger

    def training_step(self, train_loader, val_loader):
        # Training
        self.model.train()
        train_losses = []
        for q, u, dq in train_loader:
            q = q.view(q.size(0), -1).type(torch.FloatTensor).to(ptu.device)
            u = u.view(u.size(0), -1).type(torch.FloatTensor).to(ptu.device)
            dq = dq.type(torch.FloatTensor).to(ptu.device)
            self.optimizer.zero_grad()
            pred = self.model(q, u)
            loss = self.loss(pred, dq)
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss.item())
        mean_train_loss = np.mean(train_losses)

        # Validation
        self.model.eval()
        val_losses = []
        with torch.no_grad():
            for q, u, dq in val_loader:
                q = q.view(q.size(0), -1).type(torch.FloatTensor).to(ptu.device)
                u = u.view(u.size(0), -1).type(torch.FloatTensor).to(ptu.device)
                dq = dq.type(torch.FloatTensor).to(ptu.device)
                pred = self.model(q, u)
                loss = self.loss(pred, dq)
                val_losses.append(loss.item())
        mean_val_loss = np.mean(val_losses)
        self.scheduler.step()

        return {
            'train_loss': mean_train_loss,
            'val_loss': mean_val_loss,
        }

    def log_info(self, info, epoch, no_logging):
        self.logger.info(f"Epoch: {epoch}: train_loss: {info['train_loss']:.4f}, val_loss: {info['val_loss']:.4f}")
        if no_logging:
            return
        for entry, value in info.items():
            self.writer.add_scalar(entry, value, global_step=epoch)

    def training_main(self, noise_dataset, n_epochs, batch_size, **params):
        """Warning: At this point, y should already be x_{k+1} - f_nominal(x_k, u_k). """
        train_dataset, val_dataset = train_test_split(noise_dataset, test_size=0.2, shuffle=True,
                                                      random_state=params['seed'])
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        for epoch in range(n_epochs):
            info = self.training_step(train_loader, val_loader)
            self.log_info(info, epoch, params['no_logging'])

    def export(self, path='../model_data'):
        self.model.export(path, name=f'{self.behavior}_model.pkl')

    def load(self, path='../model_data'):
        self.model.load(path, name=f'{self.behavior}_model.pkl')

def get_data(behavior='noise', noise_threshold=0.1):
    dataset = None
    if behavior == 'noise':
        # pickle_file_name = f"../data/noise_dataset_old_{params['history']}.pkl"
        # if os.path.exists(pickle_file_name):
        #     with open(pickle_file_name, 'rb') as f:
        #         dataset = pkl.load(f)
        # else:
        dynamics_dataset = DynamicsDataset.from_pickle('../data/old_data.pkl', dt=params['dt'],
                                                       history=params['history'])
        dataset = NoiseDataset.from_dataset(dynamics_dataset, track, dynamics, dynamics_uses_frenet,
                                            threshold=noise_threshold)
        # with open(f"../data/noise_dataset_old_{params['history']}.pkl", 'wb') as f:
        #     pkl.dump(dataset, f)

    elif behavior == 'dynamics':
        dataset = DynamicsDataset.from_pickle('../data/old_data.pkl', dt=params['dt'],
                                              history=params['history'])
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--track_name', type=str, default='L_track_barc',
                        help='specify track (mpclab common API)')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--history', type=int, default=3, help='length of history at each step')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--n_layers', type=int, default=2, help='number of hidden layers')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--size', type=int, default=32, help='size of the hidden layers')
    parser.add_argument('--activation', type=str, default='tanh',
                        help='activation function in layers')
    parser.add_argument('--noise_threshold', type=float, default=0.1,
                        help='noise threshold to filter data')
    parser.add_argument('--output_activation', type=str, default='identity',
                        help='activation before output')
    parser.add_argument('--dt', type=float, default=0.1, help='discretization step')
    parser.add_argument('--comment', type=str, default='', help='additional comment for logs')
    parser.add_argument('--behavior', type=str, default='noise',
                        help='training on noise/dynamics')
    parser.add_argument('--no_logging', action='store_true', help='no tensorboard log')
    parser.add_argument('--no_gpu', action='store_true', help='train with CPU')

    params = vars(parser.parse_args())
    print(params)
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    ptu.init_gpu(use_gpu=not params['no_gpu'])

    # Dataset: a sequence of x_k, u_k.
    # Dataset transform:
    # 1. Use the nominal model, get x_{k + 1} - f_nominal(x_k, u_k).
    # 2.1 With history of length h, use a sliding window to concatenate h consecutive x and u to form a single input.
    # 2.2 With a sequential model, pass in the entire sequence as a single dataset. (?)
    track = mpclab_common.track.get_track('L_track_barc')
    dt = params['dt']
    dynamics_config = DynamicBicycleConfig(dt=dt,
                                           track_name='L_track_barc',
                                           mass=2.92,
                                           yaw_inertia=0.13323)
    dynamics = CasadiDynamicCLBicycle(t0=0, model_config=dynamics_config)
    dynamics_uses_frenet = True

    model = FeedforwardNoiseModel(state_size=len(q_labels), action_size=len(u_labels), **params)
    dataset = get_data(behavior=params['behavior'], noise_threshold=params['noise_threshold'])
    trainer = ModelTrainer(model, lr=params['lr'], no_logging=params['no_logging'], comment=params['comment'])
    try:
        trainer.training_main(dataset, **params)
    finally:
        trainer.export()