import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from mpclab_common.models.dynamics_models import CasadiDynamicCLBicycle
from mpclab_common.track import get_track
from mpclab_common.pytypes import VehicleState

import utils.pytorch_utils as ptu


class NoiseModel(nn.Module):
    def __init__(self, state_size, **params):
        super().__init__()
        self.state_size = state_size
        self.model: nn.Module = ...

    def forward(self, q: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        mu, tril_elem = self.model(q, u)  # mu.size=d, L.shape=d * (d + 1) / 2.
        L = torch.zeros((q.size(0), self.state_size, self.state_size))
        L[:, torch.tril_indices(self.state_size, self.state_size,
                                offset=0).tolist()] = tril_elem  # TODO: Check if this is right
        distribution = torch.distributions.MultivariateNormal(loc=mu, scale_tril=L)
        # distribution: torch.distributions.Normal = self.model(q, u)
        return distribution.rsample()

    def get_prediction(self, vehicle_state: VehicleState) -> np.ndarray:
        q, u = self.nominal_model.state2qu(vehicle_state)
        noise = ptu.to_numpy(self(ptu.from_numpy(q), ptu.from_numpy(u)))
        return noise


class CasadiDynamicCLBicycleNoise:
    def __init__(self, nominal_model: CasadiDynamicCLBicycle, noise_model: NoiseModel):
        """
        Notes:
            1. dt should be the consistent with dataset
            2. nominal_model_config should be loaded from a yaml file.
        """
        self.nominal_model = nominal_model
        self.noise_model = noise_model

    def get_nominal_prediction(self, q, u, vehicle_state=VehicleState(t=0)) -> np.ndarray:
        self.nominal_model.qu2state(vehicle_state, q, u)
        self.nominal_model.step(vehicle_state)
        return self.nominal_model.state2q(vehicle_state)

    def step(self, vehicle_state: VehicleState) -> None:
        # Before entering this function: call qu2state to fill in the q and u fields.
        noise = self.noise_model.get_prediction(vehicle_state)
        self.nominal_model.step(vehicle_state)
        q_next = self.nominal_model.state2q(vehicle_state)
        self.nominal_model.q2state(vehicle_state, q_next + noise)


if __name__ == '__main__':
    ...
