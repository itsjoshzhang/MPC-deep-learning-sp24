import os
import numpy as np
import mpclab_common
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from mpclab_common.track import get_track
from mpclab_common.pytypes import VehicleState
from mpclab_common.models.model_types import DynamicBicycleConfig
from mpclab_common.models.dynamics_models import CasadiDynamicCLBicycle

from utils.data_utils import q_local_to_global, q_global_to_local, q_labels, u_labels
from utils.log import setup_custom_logger
import utils.pytorch_utils as ptu

class FeedforwardNoiseModel(nn.Module):
    def __init__(self, state_size, action_size, history, **params):
        super().__init__()
        self.state_size = state_size
        self.model: nn.Module = ptu.build_mlp(
            input_size=(state_size + action_size) * history,
            output_size=state_size * 2,
            **params
        )
        self.model.to(ptu.device)
        self.logger = setup_custom_logger('noise_model')

    def get_logger(self):
        return self.logger

    def forward(self, q: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        input_tensor = torch.concat([q, u], dim=-1)
        output = self.model(input_tensor)
        mu, logvar = output.split(self.state_size, dim=1)
        self.logger.debug(f'mu: {mu}, logvar: {logvar}')
        distribution = torch.distributions.Normal(mu, torch.nn.functional.softplus(logvar))
        return distribution.rsample()

    def get_prediction(self, q: np.ndarray, u: np.ndarray) -> np.ndarray:
        q, u = ptu.from_numpy(q), ptu.from_numpy(u)
        # q, u = ptu.from_numpy(q).unsqueeze(1), ptu.from_numpy(u).unsqueeze(1)
        return ptu.to_numpy(self(q, u))

    def export(self, path='../model_data', name='noise_model.pkl'):
        torch.save(self.model.state_dict(), os.path.join(path, name))

    def load(self, path='../model_data', name='noise_model.pkl'):
        self.model.load_state_dict(torch.load(os.path.join(path, name)))
        self.model.to(ptu.device)

class CasadiDynamicCLBicycleNoise:
    def __init__(self,
                 nominal_model: CasadiDynamicCLBicycle,
                 noise_model: FeedforwardNoiseModel,
                 dynamics_uses_frenet: bool,
                 track=None,
                 history=1):
        self.nominal_model = nominal_model
        self.dynamics_uses_frenet = dynamics_uses_frenet
        self.track = track
        self.noise_model = noise_model
        self.history = history
        self.q_buffer = deque(maxlen=history)
        self.u_buffer = deque(maxlen=history)

    def get_nominal_prediction(self, q: np.ndarray, u: np.ndarray, vehicle_state=VehicleState(t=0)) -> np.ndarray:
        """
        q: state.v.v_long, state.v.v_tran, state.w.w_psi, state.x.x, state.x.y, state.e.psi
        u: state.u.u_a, state.u.u_steer
        Normally, you should NOT pass a vehicle_state from the outside.
        """
        assert q.shape == (len(q_labels),) and u.shape == (len(u_labels),), f"Got unexpected shape: q={q.shape}, u={u.shape}"

        if self.dynamics_uses_frenet:
            q = q_global_to_local(q, self.track)
        self.nominal_model.qu2state(vehicle_state, q, u)
        self.nominal_model.step(vehicle_state)
        nominal_next_q = self.nominal_model.state2q(vehicle_state)
        if self.dynamics_uses_frenet:
            nominal_next_q = q_local_to_global(nominal_next_q, self.track)
        return nominal_next_q

    def get_prediction(self, q, u, vehicle_state=VehicleState(t=0)) -> np.ndarray:
        """
        Does not support inference in parallel.
        q: state.v.v_long, state.v.v_tran, state.w.w_psi, state.x.x, state.x.y, state.e.psi
        u: state.u.u_a, state.u.u_steer
        Normally, you should NOT pass a vehicle_state from the outside.
        """
        assert len(q.shape) == 1 and len(u.shape) == 1

        self.q_buffer.append(q)
        self.u_buffer.append(u)
        while len(self.q_buffer) < self.history:
            self.q_buffer.append(q)
            self.u_buffer.append(u)
        nominal_next_q = self.get_nominal_prediction(q, u)
        noise = self.noise_model.get_prediction(np.asarray(self.q_buffer).reshape(1, -1),
                                                np.asarray(self.u_buffer).reshape(1, -1))
        return nominal_next_q + noise

    def step(self, state: VehicleState) -> None:
        """Wrapper for mpclab_common. """
        # Warning: We assume the global coordinates (x) are maintained.
        q = np.array([state.v.v_long, state.v.v_tran, state.w.w_psi, state.x.x, state.x.y, state.e.psi])
        u = np.array([state.u.u_a, state.u.u_steer])
        next_q = self.get_prediction(q.reshape(q.shape[0], -1), u)  # TODO: create a buffer in this class. Pass the buffer as q here.
        self.nominal_model.q2state(state, next_q)

class DynamicsNN(CasadiDynamicCLBicycleNoise):
    def get_prediction(self, q, u, vehicle_state=VehicleState(t=0)) -> np.ndarray:
        assert len(q.shape) == 1 and len(u.shape) == 1

        self.q_buffer.append(q)
        self.u_buffer.append(u)
        while len(self.q_buffer) < self.history:
            self.q_buffer.append(q)
            self.u_buffer.append(u)

        prediction = self.noise_model.get_prediction(np.asarray(self.q_buffer).reshape(1, -1),
                                                np.asarray(self.u_buffer).reshape(1, -1))
        return prediction

    def step(self, state: VehicleState) -> None:
        raise NotImplementedError

if __name__ == '__main__':
    track_name = 'L_track_barc'
    track = mpclab_common.track.get_track(track_name)
    dt = 0.1

    dynamics_config = DynamicBicycleConfig(dt=dt,
                                           track_name=track_name,
                                           mass=2.92,
                                           yaw_inertia=0.13323,
                                           )
    nominal_model = CasadiDynamicCLBicycle(t0=0, model_config=dynamics_config)
    dynamics_uses_frenet = True

    noise_model = FeedforwardNoiseModel(
        state_dim=len(q_labels),
        action_size=len(u_labels),
        dynamics_uses_frenet=True,
    )

    model = CasadiDynamicCLBicycleNoise(
        nominal_model=nominal_model,
        noise_model=noise_model,
        dynamics_uses_frenet=dynamics_uses_frenet,
        track=track
    )
    data = './data/old_data.pkl'