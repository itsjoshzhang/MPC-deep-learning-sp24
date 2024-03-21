import tqdm
import argparse
import numpy as np
import pickle as pkl
import seaborn as sns

from typing import Union
from matplotlib import pyplot as plt
from utils.log import setup_custom_logger
from torch.utils.data import Dataset, ConcatDataset

import mpclab_common.track
from mpclab_common.models.dynamics_models import CasadiDynamicBicycle, CasadiDynamicCLBicycle
from mpclab_common.models.model_types import DynamicBicycleConfig
from mpclab_common.pytypes import VehicleState

logger = setup_custom_logger('data_utils')

class DynamicsDataset(Dataset):

    def __init__(self, q, u, dt, DATASET_DT=0.02, history=1):
        
        assert q.shape[0] == u.shape[0], "q and u doesn't match in n_entries. "
        step = int(dt // DATASET_DT)
        assert np.isclose(step * DATASET_DT, dt), "dt must be a multiplier of DATASET_DT. "

        # q = q[:(q.shape[0] - (q.shape[0] % step))]
        # self.q = np.mean(q.reshape(-1, step, q.shape[1]), axis=1)
        # self.u = u[::step]

        self.q = q[::step]
        u = u[:(u.shape[0] - (u.shape[0] % step))]
        self.u = np.mean(u.reshape(-1, step, u.shape[1]), axis=1)
        self.history = history

        assert self.q.shape[0] >= self.history, (f"Dataset is too short. "
            f"Found {self.q.shape[0]} entries, while history is {self.history}. ")

    @classmethod
    def from_dict(cls, data, dt, DATASET_DT=0.02, history=1, plot=False):
        
        # data: Nested OrderedDict converted from a single rosbag. Created on 3/4/24 based on mpclab_common.
        # DATASET_DT: Based on the sampling frequency of the estimator.
        qs, us, ts = [], [], []

        for t, state in data.items():
            q = np.array(
                [state['v']['v_long'], state['v']['v_tran'], state['w']['w_psi'],
                 state['x']['x'], state['x']['y'], state['e']['psi']])
            
            u = np.array([state['u']['u_a'], state['u']['u_steer']])
            if np.allclose(u, np.array([1, 0])) or np.allclose(u, 0):
                continue
            qs.append(q)
            us.append(u)
            ts.append(t)

        t, q, u = np.asarray(ts) / 1e9, np.asarray(qs), np.asarray(us)

        # Interpolating with the standard dt. Assume t is sorted.
        t_interp = np.arange(t[0], t[-1], DATASET_DT)

        logger.debug("Points in dataset: {}; Points after interp: {}.".format(t.shape[0], t_interp.shape[0]))

        q_interp = np.empty((t_interp.shape[0], q.shape[1]))
        for i in range(q.shape[1]):
            q_interp[:, i] = np.interp(t_interp, t, q[:, i])

        u_interp = np.empty((t_interp.shape[0], u.shape[1]))
        for i in range(u.shape[1]):
            u_interp[:, i] = np.interp(t_interp, t, u[:, i])

        if plot:
            visualize_data(t_interp, q_interp, u_interp)
        return cls(q_interp, u_interp, dt, DATASET_DT, history)

    @classmethod
    def from_pickle(cls, file_name, dt, DATASET_DT=0.02, history=1, plot=False):

        with open(file_name, 'rb') as f:
            full_data = pkl.load(f)
            size = len(full_data)

        if size == 1:
            return [DynamicsDataset.from_dict(data, dt, 
                DATASET_DT, history, plot) for data in full_data.values()][0]
        else:
            return ConcatDataset([DynamicsDataset.from_dict(data, dt, 
                DATASET_DT, history, plot) for data in full_data.values()])

    def __len__(self):
        return self.q.shape[0] - self.history

    def __getitem__(self, i):
        """Returns q, u, next_q"""
        return self.q[i:i + self.history], self.u[i:i + self.history], self.q[i + self.history]


def visualize_data(t, q, u, title=''):
    q_labels = ['v.v_long', 'v.v_tran', 'w.w_psi', 'x.x', 'x.y', 'e.psi']
    u_labels = ['u.u_a', 'u.u_steer']

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


def get_nominal_prediction(dynamics, q, u, dynamics_uses_frenet, track=None,):
                           # vehicle_state=VehicleState(t=0)) -> np.ndarray:
    if dynamics_uses_frenet:
        q = q_global_to_local(q, track)

    dynamics.qu2state(vehicle_state, q, u)
    dynamics.step(vehicle_state)
    nominal_next_q = dynamics.state2q(vehicle_state)

    if dynamics_uses_frenet:
        nominal_next_q = q_local_to_global(nominal_next_q, track)
    return nominal_next_q


class NoiseDataset(Dataset):

    def __init__(self, q, u, dq, threshold=0.1):
        mask = np.linalg.norm(dq, axis=1) < threshold

        self.q = q[mask]
        self.u = u[mask]
        self.dq = dq[mask]
        assert q.shape[0] == u.shape[0] == dq.shape[0]

    @classmethod
    def from_dataset(cls, dataset: Union[DynamicsDataset, ConcatDataset],
                     track, dynamics, dynamics_uses_frenet, threshold=0.1):
        
        qs, us, dqs = [], [], []
        for q, u, next_q in tqdm(dataset):
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
        return cls(q, u, dq, threshold=threshold)

    def __len__(self):
        return self.q.shape[0]

    def __getitem__(self, idx):
        return self.q[idx], self.u[idx], self.dq[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--threshold',  type=float, default=0.1, help='Maximum allowed error')
    parser.add_argument('--dt',         type=float, default=0.1, help='Discretization step for the nominal model')
    parser.add_argument('--history',    type=int,   default=32,  help='maximum size of the buffer')
    parser.add_argument('--track_name', type=str,   default='L_track_barc', help='name of the track')
    parser.add_argument('--file_name',  type=str,   default='../data/validation/mpc_data.pkl', help='name of pickle file')
    parser.add_argument('--visualize',  action='store_true',     help='visualize data before and after')
    parser.add_argument('--load_prev',  action='store_true',     help='load previous dataset instead')

    params = vars(parser.parse_args())
    load_prev  = params['load_prev']
    visualize  = params['visualize']
    file_name  = params['file_name']
    track_name = params['track_name']
    history = params['history']
    dt = params['dt']

    # track = mpclab_common.track.get_track(params['track_name'])
    # dynamics_config = DynamicBicycleConfig(dt=dt,
    #     track_name=track_name, mass=2.92, yaw_inertia=0.13323)
    # dynamics = CasadiDynamicCLBicycle(t0=0, model_config=dynamics_config)
    # dynamics_uses_frenet = True

    if load_prev:
        with open(file_name, "rb") as f:
            noise_dataset = pkl.load(f)
    else:
        dynamics_dataset = DynamicsDataset.from_pickle(file_name, dt, history=history, plot=visualize)
        
        # noise_dataset = NoiseDataset.from_dataset(dynamics_dataset, track, dynamics,
        #     dynamics_uses_frenet=dynamics_uses_frenet, threshold=params['threshold'])
        # with open(f"../data/noise_dataset_old_{history}.pkl", "wb") as f:
        #     pkl.dump(noise_dataset, f)

    if visualize:
        x, y, z, dx, dy = [], [], [], [], []

        for q, u, dq in dynamics_dataset:
            err = np.linalg.norm((dq[3], dq[4]))
            # if err > 0.2:
            #     continue
            x.append(q[-1][3])
            y.append(q[-1][4])
            z.append(err)
            dx.append(dq[3])
            dy.append(dq[4])

        fig = plt.figure()
        ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
        ax_norm = fig.add_subplot(2, 2, 2)
        ax_x = fig.add_subplot(2, 2, 3)
        ax_y = fig.add_subplot(2, 2, 4)

        ax_3d.scatter(x, y, z, c='k', linestyle='-', marker='.')
        ax_3d.set_title('Distribution of l-2 norm of prediction error')
        ax_3d.set_xlabel('x (m)')
        ax_3d.set_ylabel('y (m)')
        ax_3d.set_zlabel('error (m)')

        hist = sns.histplot(data=z, ax=ax_norm)
        hist.set(xlabel='l-2 norm of error')
        labels = [str(v) if v else '' for v in hist.containers[0].datavalues]
        hist.bar_label(hist.containers[0], label=labels)
        ax_norm.set_title("Error Norm Distribution")

        hist_x = sns.histplot(data=dx, ax=ax_x)
        hist_x.set(xlabel='x error')
        labels = [str(v) if v else '' for v in hist_x.containers[0].datavalues]
        hist_x.bar_label(hist_x.containers[0], label=labels)
        ax_x.set_title("X-Error Distribution")

        hist_y = sns.histplot(data=dy, ax=ax_y)
        hist_y.set(xlabel='y error')
        labels = [str(v) if v else '' for v in hist_y.containers[0].datavalues]
        hist_y.bar_label(hist_y.containers[0], label=labels)
        ax_y.set_title("Y-Error Distribution")
        plt.show()