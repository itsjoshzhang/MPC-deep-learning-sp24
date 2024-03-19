import os
import torch
import argparse
import numpy as np

from matplotlib import pyplot as plt
from mpclab_common.track import get_track
from mpclab_common.models.dynamics_models import DynamicBicycleConfig, CasadiDynamicCLBicycle

from model_scripts.noise_model import FeedforwardNoiseModel, CasadiDynamicCLBicycleNoise, DynamicsNN
from utils.data_utils import DynamicsDataset, NoiseDataset, q_labels, u_labels
from utils.log import setup_custom_logger

logger = setup_custom_logger('test')


def prepare_components(history,
                       track_name='L_track_barc',
                       dt=0.1,
                       size=32,
                       n_layers=2,
                       activation='tanh',
                       output_activation='identity',
                       behavior='noise',
                       **params):
    track = get_track(track_name)

    dynamics_config = DynamicBicycleConfig(dt=dt,
                                           track_name=track_name,
                                           mass=2.92,
                                           yaw_inertia=0.13323,
                                           )
    dynamics = CasadiDynamicCLBicycle(t0=0, model_config=dynamics_config)
    dynamics_uses_frenet = True

    noise_model = FeedforwardNoiseModel(state_size=len(q_labels), action_size=len(u_labels), history=history,
                                        size=size, n_layers=n_layers, activation=activation, output_activation=output_activation
                                        )
    noise_model.load(name=f'{behavior}_model.pkl')
    noise_model.eval()

    if behavior == 'noise':
        model = CasadiDynamicCLBicycleNoise(nominal_model=dynamics,
                                            noise_model=noise_model,
                                            dynamics_uses_frenet=dynamics_uses_frenet,
                                            track=track,
                                            history=history)
    elif behavior == 'dynamics':
        model = DynamicsNN(nominal_model=dynamics,
                           noise_model=noise_model,
                           dynamics_uses_frenet=dynamics_uses_frenet,
                           track=track,
                           history=history)
    else:
        raise NotImplementedError

    return model


def prepare_dataset(file_name, dt=0.1, **params):
    full_name = os.path.join('../data', f'{file_name}_data.pkl')
    if not os.path.exists(full_name):
        raise FileNotFoundError(full_name)
    return DynamicsDataset.from_pickle(full_name, dt=dt, history=1)


def run_dynamics_model(model, dynamics_dataset, open_loop: bool):
    q_data, q_nominal, q_noised = [], [], []
    labels = ['data', 'nominal', 'with noise']

    for step, (q, u, _) in enumerate(dynamics_dataset):
        try:
            q_data.append(q[-1])

            if len(q_nominal) == 0:
                q_nominal.append(q[-1])
            else:
                q_nominal.append(model.get_nominal_prediction(q_nominal[-1] if open_loop else q[-1], u[-1]))

            if len(q_noised) == 0:
                q_noised.append(q[-1])
            else:
                q_noised.append(model.get_prediction(q_noised[-1] if open_loop else q[-1], u[-1]).squeeze(0))
        except ValueError as e:
            logger.error(f"'{e}' at step {step}.")
            break
    return (q_data, q_nominal, q_noised), labels


def visualize_trajectories(trajectories, labels, save_plots, *args):
    fig, ax = plt.subplots()
    for trajectory, label in zip(trajectories, labels):
        data_array = np.asarray(trajectory)
        logger.debug(data_array.shape)
        ax.plot(data_array[:, 3], data_array[:, 4], label=label)

    ax.legend()
    if save_plots:
        plt.savefig(f'../plots/{"_".join([str(x) for x in args])}.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--file_name', type=str, help='pickle file name stem for rosbag dict')
    parser.add_argument('--dt', type=float, default=0.1, help='Discretization timestep for the dynamics')
    parser.add_argument('--history', type=int, default=3, help='length of history at each step')
    parser.add_argument('--track_name', type=str, default='L_track_barc',
                        help='name of the track (mpclab_common API)')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--n_layers', type=int, default=2, help='number of hidden layers')
    parser.add_argument('--size', type=int, default=32, help='size of the hidden layers')
    parser.add_argument('--activation', type=str, default='tanh',
                        help='activation function in layers')
    parser.add_argument('--output_activation', type=str, default='identity',
                        help='activation before output')
    parser.add_argument('--open_loop', action='store_true', help='Do open loop prediction instead')
    parser.add_argument('--save_plots', action='store_true', help='save plots to plots folder')
    parser.add_argument('--behavior', type=str, default='noise',
                        help='model trained on noise/dynamics')

    params = vars(parser.parse_args())
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    dataset = prepare_dataset(**params)
    model = prepare_components(**params)
    trajectories, labels = run_dynamics_model(model, dataset, params['open_loop'])
    visualize_trajectories(trajectories, labels, params['save_plots'], params['file_name'])