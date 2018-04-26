from train_mefn import train_mefn
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import os
from efn_util import plotMefnTraining

os.chdir('../');

exp_fam = 'dirichlet';
D = 3;
flow_id = 'planar2';
cost_type = 'KL';
M_eta = 1000;
lr_order = -2;
random_seed = 2;

alphas = [];
alphas.append(np.array([1,2,3]));
#alphas.append(np.array([1, 2, 3]));
#alphas.append(np.array([3, 2, 1]));
#alphas.append(np.array([1, 2, 2, 2, 1]));
#alphas.append(np.array([3, 2, 1, 1, 3]));

check_rate = 100;
num_alphas = len(alphas);
for i in range(num_alphas):
	#alpha = alphas[i];
	alpha = np.expand_dims(alphas[i], 0);
	D = alpha.shape[1];
	params = {'alpha': alpha, 'D':D};
	log_P, X, R2s, KLs, it = train_mefn(exp_fam, params, flow_id, cost_type, M_eta, lr_order, random_seed, check_rate);
	print('xshape', X.shape);
	print('learned in %d iterations' % it)
	plotMefnTraining(exp_fam, R2s, KLs, X, log_P, params, check_rate, it, 'dirichlet %d' % (i+1));
plt.show();


