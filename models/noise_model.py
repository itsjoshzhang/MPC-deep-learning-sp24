import copy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from mpclab_common.models.dynamics_models import CasadiDynamicCLBicycle
from mpclab_common.track import get_track
from mpclab_common.pytypes import VehicleState
from utils.data_utils import q_local_to_global, q_global_to_local

import utils.pytorch_utils as ptu


class NoiseModel(nn.Module):
    def __init__(self, state_size, action_size, history, **params):
        super().__init__()
        self.state_size = state_size
        self.model: nn.Module = ptu.build_mlp(
            input_size=(state_size + action_size) * history,
            output_size=state_size * 2,
            **params
        )
        self.model.to(ptu.device)

    def forward(self, q: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        input_tensor = torch.concat([q, u], dim=-1)
        output = self.model(input_tensor)
        mu, logvar = output.split(self.state_size, dim=1)
        distribution = torch.distributions.Normal(mu, torch.exp(logvar))
        return distribution.rsample()

    def get_prediction(self, q: np.ndarray, u: np.ndarray) -> np.ndarray:
        q, u = ptu.from_numpy(q).unsqueeze(1), ptu.from_numpy(u).unsqueeze(1)
        return ptu.to_numpy(self(q, u))


class CasadiDynamicCLBicycleNoise:
    def __init__(self,
                 nominal_model: CasadiDynamicCLBicycle,
                 noise_model: NoiseModel,
                 dynamics_uses_frenet: bool,
                 track=None):
        self.nominal_model = nominal_model
        self.dynamics_uses_frenet = dynamics_uses_frenet
        self.track = track
        self.noise_model = noise_model

    def get_nominal_prediction(self, q, u, vehicle_state=VehicleState(t=0)) -> np.ndarray:
        if self.dynamics_uses_frenet:
            q = q_global_to_local(q, self.track)
        self.nominal_model.qu2state(vehicle_state, q, u)
        self.nominal_model.step(vehicle_state)
        nominal_next_q = self.nominal_model.state2q(vehicle_state)
        if self.dynamics_uses_frenet:
            nominal_next_q = q_local_to_global(nominal_next_q, self.track)
        return nominal_next_q

    def step(self, state: VehicleState) -> None:
        # Warning: We assume the global coordinates (x) are maintained.
        q = np.array([state.v.v_long, state.v.v_tran, state.w.w_psi, state.x.x, state.x.y, state.e.psi])
        u = np.array([state.u.u_a, state.u.u_steer])
        if self.dynamics_uses_frenet:
            self.track.global_to_local_typed(state)
        noise = self.noise_model.get_prediction(q, u)
        self.nominal_model.step(state)
        q_next = self.nominal_model.state2q(state)
        self.nominal_model.q2state(state, q_next + noise)


if __name__ == '__main__':
    ...
