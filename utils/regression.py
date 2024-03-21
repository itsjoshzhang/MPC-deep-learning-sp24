import argparse
import numpy as np
import data_utils as DU
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.stats import norm

PARAMS = ['v_long', 'v_tran', 'w_psi', 'x_pos',
          'y_pos', 'e_psi',  'u_a', 'u_steer']

class DynamicsRegression:

    def __init__(self, dataset):
        self.dataset = dataset

        q, u, _ = dataset[0]
        self.q_size = q.shape[1]
        self.u_size = u.shape[1]
        self.qu = self.q_size + self.u_size

        self.slopes = np.zeros((len(dataset), self.qu))   # Save regressed parameters
        self.inters = np.zeros((len(dataset), self.qu))   # (state=6) + (input=2) = 8

    def local_regression(self):
        time = np.arange(self.dataset.history) # Assuming equal? timesteps
        
        for i in range(len(self.dataset)):
            q, u, _ = self.dataset[i]
            s = self.q_size

            for j in range(s):            # 6 state parameters
                slope, inter,_,_,_ = linregress(time, q[:, j])
                self.slopes[i, j] = slope
                self.inters[i, j] = inter

            for j in range(self.u_size):  # 2 input parameters
                slope, inter,_,_,_ = linregress(time, u[:, j])
                self.slopes[i, j + s] = slope
                self.inters[i, j + s] = inter

    def total_regression(self):
        time = np.arange(self.slopes.shape[0])  # Assuming equal? timesteps
       
        total_slopes = np.zeros(self.qu)
        total_inters = np.zeros(self.qu)

        for i in range(self.qu):
            slope, inter,_,_,_ = linregress(time, self.slopes[:, i])
            total_slopes[i] = slope
            total_inters[i] = inter
    
        return total_slopes, total_inters
    
    def visualize(self):
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Slopes')

        for i in range(self.qu):
            row, col = divmod(i, 4)
            axs[row, col].plot(np.arange(len(self.dataset)), 
                                self.slopes[:, i], label='Slope')
            
            axs[row, col].set_title(PARAMS[i])
            axs[row, col].set_xlabel('Time')
            axs[row, col].set_ylabel('Slope')
        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Intercepts')

        for i in range(self.qu):
            row, col = divmod(i, 4)
            axs[row, col].plot(np.arange(len(self.dataset)), 
                self.inters[:, i], label='Intercept', color='orange')
            
            axs[row, col].set_title(PARAMS[i])
            axs[row, col].set_xlabel('Time')
            axs[row, col].set_ylabel('Intercept')
        plt.tight_layout()
        plt.show()
    
    def run(self):
        self.local_regression()
        self.total_regression()
        self.visualize()


class DynamicsGaussian:

    def __init__(self, dataset):
        self.dataset = dataset

        q, u, _ = dataset[0]
        self.q_size = q.shape[1]
        self.u_size = u.shape[1]
        self.qu = self.q_size + self.u_size

        # Instead of slopes and intercepts, we save means and standard deviations
        self.means = np.zeros((len(dataset), self.qu))
        self.stds = np.zeros((len(dataset), self.qu))

    def local_gaussian_fit(self):
        for i in range(len(self.dataset)):
            q, u, _ = self.dataset[i]
            s = self.q_size

            # Fit Gaussian for each state parameter
            for j in range(s):
                mean, std = norm.fit(q[:, j])
                self.means[i, j] = mean
                self.stds[i, j] = std

            # Fit Gaussian for each input parameter
            for j in range(self.u_size):
                mean, std = norm.fit(u[:, j])
                self.means[i, j + s] = mean
                self.stds[i, j + s] = std

    def total_gaussian_fit(self):
        # Aggregate means and stds over all groups
        total_means = np.zeros(self.qu)
        total_stds = np.zeros(self.qu)

        for i in range(self.qu):
            mean, std = norm.fit(self.means[:, i])
            total_means[i] = mean
            total_stds[i] = std

        return total_means, total_stds

    def visualize(self):
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Means')

        for i in range(self.qu):
            row, col = divmod(i, 4)
            axs[row, col].plot(np.arange(len(self.dataset)), 
                               self.means[:, i], label='Mean')
            
            axs[row, col].set_title(PARAMS[i])
            axs[row, col].set_xlabel('Time')
            axs[row, col].set_ylabel('Mean')
        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Stdevs')

        for i in range(self.qu):
            row, col = divmod(i, 4)
            axs[row, col].plot(np.arange(len(self.dataset)), 
                               self.stds[:, i], label='Stdev', color='orange')
            
            axs[row, col].set_title(PARAMS[i])
            axs[row, col].set_xlabel('Time')
            axs[row, col].set_ylabel('Stdev')
        plt.tight_layout()
        plt.show()

    def run(self):
        self.local_gaussian_fit()
        total_means, total_stds = self.total_gaussian_fit()
        print("Total Means:", total_means)
        print("Total Stdevs:", total_stds)
        self.visualize()
        

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

    dynamics_dataset = DU.DynamicsDataset.from_pickle(file_name, dt, history=history, plot=visualize)
    dynamics_regression = DynamicsRegression(dynamics_dataset)
    dynamics_regression.run()