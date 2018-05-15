from train_efn import train_efn
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import os, sys
from efn_util import get_flowdict, print_flowdict

os.chdir('../');

exp_fam = 'normal';
D = int(sys.argv[1]);
planar_layers = int(sys.argv[2]);

flow_dict = get_flowdict(0, planar_layers, 0, 0);
flow_ids = flow_dict['flow_ids'];
flow_repeats = flow_dict['flow_repeats'];
print_flowdict(flow_dict);

cost_type = 'KL';
K_eta = 100;
M_eta = 100;
stochastic_eta = True;
L = 8;
upl_tau = 0.5;
give_inverse_hint = True;
lr_order = -3;
random_seed = 0;
max_iters = 10000;
check_rate = 100;

X, train_KLs, it = train_efn(exp_fam, D, flow_dict, cost_type, K_eta, M_eta, stochastic_eta, \
	                         give_inverse_hint, lr_order, random_seed, max_iters, check_rate);
