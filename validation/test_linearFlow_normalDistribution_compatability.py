from train_network import train_network
import numpy as np
from matplotlib import pyplot as plt
from efn_util import load_constraint_info
from scipy.stats import multivariate_normal
import os

os.chdir('../');

plot_dists = True;
numdists = 125;
flow_id = 'linear1';
cost_type = 'KL';
L = 1;
n = 300;
K_eta = None;
stochastic_eta = False;
param_network = False;
lr_order = -3;
random_seed = 0;

for i in range(0, numdists):
	constraint_id = 'normal_%d' % (i+1);
	D_Z, K_eta_params, params, constraint_type = load_constraint_info(constraint_id);
	X, R2s, it = train_network(constraint_id, flow_id, cost_type,  L, n, K_eta, \
                           stochastic_eta, param_network, lr_order, random_seed);
	if (plot_dists):
		mu = params['mu_targs'][0];
		Sigma = params['Sigma_targs'][0];
		X_true = np.random.multivariate_normal(mu, Sigma, n);
		print(X_true.shape);
		plt.figure();
		plt.scatter(X[:,0,0], X[:,1,0], color=[0,0,.8]);
		plt.scatter(X_true[:,0], X_true[:,1], color=[.8,0,0]);
		plt.xlabel('x_1');
		plt.ylabel('x_2');
		plt.legend(['f_theta', 'target']);
		plt.title('Sigma=[%.2f, %.2f; %.2f, %.2f], R2 = %.4f' % \
			       (Sigma[0,0], Sigma[0,1], Sigma[1,0], Sigma[1,1], R2s[0]));
		plt.show();