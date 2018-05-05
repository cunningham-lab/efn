from train_nf import train_nf
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import os, sys
from efn_util import get_flowdict, print_flowdict

os.chdir('../');

exp_fam = 'dirichlet';
D = int(sys.argv[1]);
fully_connected_layers = int(sys.argv[2]);
planar_layers = int(sys.argv[3]);
spinner_layers = int(sys.argv[4]);
nonlin_spinner_layers = int(sys.argv[5]);
cost_type = 'KL';
M_eta = 1000;
lr_order = -3;
random_seed = 0;

flow_dict = get_flowdict(fully_connected_layers, planar_layers, spinner_layers, nonlin_spinner_layers);
print_flowdict(flow_dict);

alphas = [];
vec = np.random.uniform(0.5, 4, (D,));
alphas.append(vec);

check_rate = 100;
max_iters = 10000;
num_alphas = len(alphas);
for i in range(num_alphas):
	alpha = np.expand_dims(alphas[i], 0);
	D = alpha.shape[1];
	params = {'alpha': alpha, 'D':D};
	log_P, X, R2s, KLs, it = train_nf(exp_fam, params, flow_dict, cost_type, M_eta, lr_order, random_seed, max_iters, check_rate);
	print('xshape', X.shape);
	print('learned in %d iterations' % it);

