import random

import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset, ConcatDataset

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
        assert self.q.shape[0] >= self.history, (f"Dataset is too short. "
                                                 f"Found {self.q.shape[0]} entries, while history is {self.history}. ")

    @classmethod
    def from_dict(cls, data, dt, DATASET_DT=0.02, history=1):
        """
            data: Nested OrderedDict converted from a single rosbag. Created on 3/4/24 based on mpclab_common.
            DATASET_DT: Based on the sampling frequency of the estimator.
            """
        qs, us, ts = [], [], []
        for t, state in data.items():
            q = np.array(
                [state['v']['v_long'], state['v']['v_tran'], state['w']['w_psi'], state['x']['x'], state['x']['y'],
                 state['e']['psi']])
            # q = np.array([state['v']['v_long'], state['v']['v_tran'], state['w']['w_psi'],
            #               state['x']['x'], state['x']['y'], state['x']['z']])
            u = np.array([state['u']['u_a'], state['u']['u_steer']])
            if np.allclose(u, np.array([1, 0])) or np.allclose(u, 0):
                continue
            qs.append(q)
            us.append(u)
            ts.append(t)
        t, q, u = np.asarray(ts) / 1e9, np.asarray(qs), np.asarray(us)

        # Interpolating with the standard dt.
        t_interp = np.arange(t[0], t[-1], DATASET_DT)  # Assume t is sorted.

        print("Points in dataset: {}; Points after interp: {}.".format(t.shape[0], t_interp.shape[0]))

        q_interp = np.empty((t_interp.shape[0], q.shape[1]))
        for i in range(q.shape[1]):
            q_interp[:, i] = np.interp(t_interp, t, q[:, i])

        u_interp = np.empty((t_interp.shape[0], u.shape[1]))
        for i in range(u.shape[1]):
            u_interp[:, i] = np.interp(t_interp, t, u[:, i])

        # visualize_data(t_interp, q_interp, u_interp)
        return cls(q_interp, u_interp, dt=dt, history=history, DATASET_DT=DATASET_DT)

    @classmethod
    def from_pickle(cls, file_name, dt, DATASET_DT=0.02, history=1):
        with open(file_name, 'rb') as f:
            full_data = pkl.load(f)
        return ConcatDataset(
            [DynamicsDataset.from_dict(data, dt=dt, DATASET_DT=DATASET_DT, history=history) for data in
             full_data.values()]
        )

    def __len__(self):
        return self.q.shape[0] - self.history

    def __getitem__(self, idx):
        """Returns q, u, next_q"""
        return self.q[idx:idx + self.history], self.u[idx:idx + self.history], self.q[idx + self.history]


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


def q_global_to_local(q_global, track):
    q_local = np.copy(q_global)
    q_local[np.array([4, 5, 3])] = track.global_to_local(q_local[3:])
    return q_local


def q_local_to_global(q_local, track):
    q_global = np.copy(q_local)
    q_global[3:] = track.local_to_global(q_global[np.array([4, 5, 3])])
    return q_global


def get_nominal_prediction(dynamics, q, u, dynamics_uses_frenet, track=None,
                           vehicle_state=VehicleState(t=0)) -> np.ndarray:
    if dynamics_uses_frenet:
        q = q_global_to_local(q, track)
    dynamics.qu2state(vehicle_state, q, u)
    dynamics.step(vehicle_state)
    nominal_next_q = dynamics.state2q(vehicle_state)
    if dynamics_uses_frenet:
        nominal_next_q = q_local_to_global(nominal_next_q, track)
    return nominal_next_q


class NoiseDataset(Dataset):
    def __init__(self, q, u, dq):
        self.q = q
        self.u = u
        self.dq = dq
        assert q.shape[0] == u.shape[0] == dq.shape[0]

    @classmethod
    def from_dataset(cls, dataset: DynamicsDataset, track, dynamics, dynamics_uses_frenet):
        qs, us, dqs = [], [], []
        for q, u, next_q in dataset:
            last_q, last_u = q[-1], u[-1]
            try:
                nominal_next_q = get_nominal_prediction(dynamics, last_q, last_u, dynamics_uses_frenet, track)
            except ValueError as e:
                continue
            dq = next_q - nominal_next_q
            qs.append(q)
            us.append(u)
            dqs.append(dq)

        q, u, dq = np.asarray(qs), np.asarray(us), np.asarray(dqs)
        return cls(q, u, dq)

    def __len__(self):
        return self.q.shape[0]

    def __getitem__(self, idx):
        return self.q[idx], self.u[idx], self.dq[idx]


if __name__ == '__main__':
    track_name = 'L_track_barc'
    track = mpclab_common.track.get_track(track_name)
    dt = 0.1

    dynamics_config = DynamicBicycleConfig(dt=dt,
                                           track_name=track_name,
                                           mass=2.92,
                                           yaw_inertia=0.13323,
                                           )
    dynamics = CasadiDynamicCLBicycle(t0=0, model_config=dynamics_config)
    dynamics_uses_frenet = True

    # dynamics = CasadiDynamicBicycle(t0=0, model_config=dynamics_config)
    # dynamics_uses_frenet = False

    # file_name = '../data/pid_data.pkl'
    file_name = '../data/mpc_data.pkl'
    file_name = '../data/old_data.pkl'
    dynamics_dataset = DynamicsDataset.from_pickle(file_name, dt=dt, history=3)
    noise_dataset = NoiseDataset.from_dataset(dynamics_dataset, track, dynamics,
                                              dynamics_uses_frenet=dynamics_uses_frenet)
    with open("../data/noise_dataset_old_3.pkl", "wb") as f:
        pkl.dump(noise_dataset, f)
    x, y, z = [], [], []
    for q, u, dq in noise_dataset:
        err = np.linalg.norm((dq[3], dq[4]))
        # if err > 0.2:
        #     continue
        x.append(q[-1][3])
        y.append(q[-1][4])
        z.append(err)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, c='k', linestyle='-', marker='.')
    ax.set_title('Distribution of l-2 norm of prediction error')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('error (m)')
    plt.show()
