import copy
import random

import numpy as np
from collections import OrderedDict
import pickle as pkl
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset

from mpclab_common.models.dynamics_models import CasadiDynamicCLBicycle
from models.noise_model import CasadiDynamicCLBicycleNoise, NoiseModel

from models.noise_model import NoiseModel, CasadiDynamicCLBicycleNoise
from mpclab_common.models.dynamics_models import CasadiDynamicBicycle, CasadiDynamicCLBicycle
from mpclab_common.models.model_types import DynamicBicycleConfig
from mpclab_common.pytypes import VehicleState
import mpclab_common.track

q_labels = ['v.v_long', 'v.v_tran', 'w.w_psi', 'x.x', 'x.y', 'e.psi']
# q_labels = ['v.v_long', 'v.v_tran', 'w.w_psi', 'x.x', 'x.y', 'x.z']
u_labels = ['u.u_a', 'u.u_steer']


class DynamicsDataset(Dataset):
    def __init__(self, q, u, dt, *, DATASET_DT=0.02, history=1):
        assert q.shape[0] == u.shape[0], "q and u doesn't match in n_entries. "
        step = int(dt // DATASET_DT)
        assert np.isclose(step * DATASET_DT, dt), "dt must be a multiplier of DATASET_DT. "

        self.q = q[::step]
        # q = q[:(q.shape[0] - (q.shape[0] % step))]
        # self.q = np.mean(q.reshape(-1, step, q.shape[1]), axis=1)

        # self.u = u[::step]
        u = u[:(u.shape[0] - (u.shape[0] % step))]
        self.u = np.mean(u.reshape(-1, step, u.shape[1]), axis=1)
        self.history = history

    def __len__(self):
        return self.q.shape[0] - self.history

    def __getitem__(self, idx):
        """Returns q, u, next_q"""
        return self.q[idx:idx + self.history], self.u[idx:idx + self.history], self.q[idx + self.history]


def parse_data(data, DATASET_DT=0.02, **kwargs):
    """
    data: Nested OrderedDict converted from a single rosbag. Created on 3/4/24 based on mpclab_common.
    DATASET_DT: Based on the sampling frequency of the estimator.
    """
    # assert isinstance(data, OrderedDict), "Expecting OrderedDict, got {}".format(type(data))
    qs, us, ts = [], [], []
    for t, state in data.items():
        q = np.array([state['v']['v_long'], state['v']['v_tran'], state['w']['w_psi'], state['x']['x'], state['x']['y'], state['e']['psi']])
        # q = np.array([state['v']['v_long'], state['v']['v_tran'], state['w']['w_psi'],
        #               state['x']['x'], state['x']['y'], state['x']['z']])
        u = np.array([state['u']['u_a'], state['u']['u_steer']])
        if np.allclose(u, np.array([1, 0])) or np.allclose(u, 0):
            continue
        ts.append(t)
        qs.append(q)
        us.append(u)
    t, q, u = np.asarray(ts) / 1e9, np.asarray(qs), np.asarray(us)

    # Interpolating with the standard dt.
    t_interp = np.arange(np.ceil(t[0]), np.floor(t[-1]), DATASET_DT)  # Assume t is sorted.

    print("Points in dataset: {}; Points after interp: {}.".format(t.shape[0], t_interp.shape[0]))

    q_interp = np.empty((t_interp.shape[0], q.shape[1]))
    for i in range(q.shape[1]):
        q_interp[:, i] = np.interp(t_interp, t, q[:, i])

    u_interp = np.empty((t_interp.shape[0], u.shape[1]))
    for i in range(u.shape[1]):
        u_interp[:, i] = np.interp(t_interp, t, u[:, i])

    # visualize_data(t_interp, q_interp, u_interp)
    return DynamicsDataset(q_interp, u_interp, **kwargs)


def visualize_data(t, q, u, title=''):
    n = np.ceil(np.sqrt(len(q_labels) + len(u_labels))).astype(int)
    fig, axes = plt.subplots(n, n, sharex=True)
    fig.suptitle(title)
    fig.tight_layout()

    for i, label in enumerate(q_labels):
        ax = axes[i // n, i % n]
        ax.plot(t, q[:, i], label=label)
        ax.set_title(label)

    for i, label in enumerate(u_labels, start=len(q_labels)):
        ax = axes[i // n, i % n]
        ax.plot(t, u[:, i - len(q_labels)], label=label)
        ax.set_title(label)

    plt.show()


def get_nominal_prediction(dynamics, q, u, vehicle_state=VehicleState(t=0)) -> np.ndarray:
    dynamics.qu2state(vehicle_state, q, u)
    dynamics.step(vehicle_state)
    return dynamics.state2q(vehicle_state)


class NoiseDataset(Dataset):
    def __init__(self, q, u, dq):
        self.q = q
        self.u = u
        self.dq = dq
        assert q.shape[0] == u.shape[0] == dq.shape[0]

    @classmethod
    def from_dataset(cls, dataset: DynamicsDataset, track, dynamics, dynamics_uses_frenet):
        qs = []
        us = []
        dqs = []
        for q, u, next_q in dataset:
            qs.append(copy.deepcopy(q))
            us.append(copy.deepcopy(u))
            last_q, last_u = q[-1], u[-1]
            if dynamics_uses_frenet:
                last_q[np.array([4, 5, 3])] = track.global_to_local(last_q[3:])

            nominal_next_q = get_nominal_prediction(dynamics, last_q, last_u)
            if dynamics_uses_frenet:
                nominal_next_q[3:] = track.local_to_global(nominal_next_q[np.array([4, 5, 3])])

            dq = next_q - nominal_next_q
            dqs.append(dq)

        q, u, dq = np.asarray(qs), np.asarray(us), np.asarray(dqs)
        return cls(q, u, dq)

    def __len__(self):
        return self.q.shape[0]

    def __getitem__(self, idx):
        return self.q[idx], self.u[idx], self.dq[idx]


if __name__ == '__main__':
    track = mpclab_common.track.get_track('L_track_barc')

    dt = 0.1

    dynamics_config = DynamicBicycleConfig(dt=dt,
                                           track_name='L_track_barc',
                                           mass=2.92,
                                           yaw_inertia=0.13323,
                                           # yaw_inertia=0.033,
                                           )
    dynamics = CasadiDynamicCLBicycle(t0=0, model_config=dynamics_config)
    dynamics_uses_frenet = True

    # dynamics = CasadiDynamicBicycle(t0=0, model_config=dynamics_config)
    # dynamics_uses_frenet = False

    # file_name = '../data/pid_data.pkl'
    file_name = '../data/mpc_data.pkl'
    with open(file_name, 'rb') as f:
        full_data = pkl.load(f)
    # As the test script, we only randomly select one rosbag from the full dataset.
    rosbag = random.choice(list(full_data.keys()))
    data = full_data[rosbag]
    dynamics_dataset = parse_data(data, dt=dt, history=3)
    noise_dataset = NoiseDataset.from_dataset(dynamics_dataset, track, dynamics,
                                              dynamics_uses_frenet=dynamics_uses_frenet)
    x, y, z = [], [], []
    for q, u, dq in noise_dataset:
        x.append(q[-1][3])
        y.append(q[-1][4])
        z.append(np.linalg.norm((dq[3], dq[4])))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(x, y, z, c='k', linestyle='-', marker='.')
    ax.set_title('Distribution of l-2 norm of prediction error')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('error (m)')
    plt.show()
