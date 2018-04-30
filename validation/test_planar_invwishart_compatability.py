from train_mefn import train_mefn
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import os
from plot_utils import plotMefnTraining

os.chdir('../');

exp_fam = 'inv_wishart';
D = 5;
flow_id = 'planar20';
cost_type = 'KL';
M_eta = 1000;
lr_order = -2;
random_seed = 0;

ms = [10];
Psis = [];
Psi = np.eye(D);
Psis.append(Psi);
#alphas.append(np.array([1, 2, 3]));
#alphas.append(np.array([3, 2, 1]));
#alphas.append(np.array([1, 2, 2, 2, 1]));
#alphas.append(np.array([3, 2, 1, 1, 3]));

max_iters = 10000;
check_rate = 50;
num_Psis = len(Psis);
for i in range(num_Psis):
	#alpha = alphas[i];
	Psi = np.expand_dims(Psis[i], 0);
	m = np.array([ms[i]]);
	D = Psi.shape[1]**2;
	params = {'Psi': Psi, 'm':m, 'D':D};
	log_P, X, R2s, KLs, it = train_mefn(exp_fam, params, flow_id, cost_type, \
		                                M_eta, lr_order, random_seed, max_iters, check_rate);
	print('xshape', X.shape);
	print('took %d iterations' % it)
	#plotMefnTraining(exp_fam, R2s, KLs, X, log_P, params, check_rate, it, 'dirichlet %d' % (i+1));
plt.show();


