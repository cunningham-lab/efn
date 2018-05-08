from train_efn import train_efn
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import os, sys
from efn_util import get_flowdict, print_flowdict

os.chdir('../');

exp_fam = str(sys.argv[1]);
D = int(sys.argv[2]);
K_eta = int(sys.argv[3]);
fully_connected_layers = int(sys.argv[4]);
planar_layers = int(sys.argv[5]);
spinner_layers = int(sys.argv[6]);
nonlin_spinner_layers = int(sys.argv[7]);
stochastic_eta = int(sys.argv[8]) == 1;
give_inverse_hint = int(sys.argv[9]) == 1;
random_seed = int(sys.argv[10]);

flow_dict = get_flowdict(fully_connected_layers, planar_layers, spinner_layers, nonlin_spinner_layers);
flow_ids = flow_dict['flow_ids'];
flow_repeats = flow_dict['flow_repeats'];
print_flowdict(flow_dict);

cost_type = 'KL';
M_eta = 100;
L = 8;
upl_tau = 0.5;
lr_order = -3;
max_iters = 50000;
check_rate = 100;

X, train_KLs, it = train_efn(exp_fam, D, flow_dict, cost_type, K_eta, M_eta, stochastic_eta, \
	                       L, upl_tau, give_inverse_hint, lr_order, random_seed, max_iters, check_rate);
