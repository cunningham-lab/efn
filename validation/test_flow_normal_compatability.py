from train_nf import train_nf
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import os
import sys
from efn_util import drawEtas
from plot_utils import plotMefnTraining
from efn_util import get_flowdict, print_flowdict

os.chdir('../');

exp_fam = 'normal';
D = int(sys.argv[1]);
fully_connected_layers = int(sys.argv[2]);
planar_layers = int(sys.argv[3]);
spinner_layers = int(sys.argv[4]);
nonlin_spinner_layers = int(sys.argv[5]);

flow_dict = get_flowdict(fully_connected_layers, planar_layers, spinner_layers, nonlin_spinner_layers);
print_flowdict(flow_dict);

cost_type = 'KL';
M_eta = 1000;
give_inverse_hint = False;
lr_order = -3;
random_seed = 0;
check_rate = 100;
max_iters = 20000;
ndraws = 1;

np.random.seed(0);
for i in range(ndraws):
	etas, param_net_input, params = drawEtas(exp_fam, D, 1, give_inverse_hint);
	log_P, X, R2s, KLs, it = train_nf(exp_fam, params, flow_dict, cost_type, \
		                              M_eta, lr_order, \
		                              random_seed, max_iters, check_rate);
	#plotMefnTraining(exp_fam, R2s, KLs, X, log_P, params, check_rate, it, 'normal %d' % (i+1));


