from train_mefn import train_mefn
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import os
from efn_util import plotMefnTraining, drawEtas

os.chdir('../');

exp_fam = 'normal';
D = 2;
flow_id = 'linear1';
cost_type = 'KL';
M_eta = 1000;
lr_order = -2;
random_seed = 0;
check_rate = 100;
max_iters = 3000;
ndraws = 1;

np.random.seed(0);
for i in range(ndraws):
	etas, params = drawEtas(exp_fam, D, 1);
	print(etas);
	print(params);
	log_P, X, R2s, KLs, it = train_mefn(exp_fam, params, flow_id, cost_type, M_eta, lr_order, random_seed, max_iters, check_rate);
	print('xshape', X.shape);
	print('learned in %d iterations' % it)
	plotMefnTraining(exp_fam, R2s, KLs, X, log_P, params, check_rate, it, 'normal %d' % (i+1));
plt.show();


