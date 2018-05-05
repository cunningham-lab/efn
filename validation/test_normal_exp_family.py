from train_efn import train_efn
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import os, sys
from efn_util import get_flowdict, print_flowdict

os.chdir('../');

exp_fam = 'normal';
D = int(sys.argv[1]);
fully_connected_layers = int(sys.argv[2]);
planar_layers = int(sys.argv[3]);
spinner_layers = int(sys.argv[4]);
nonlin_spinner_layers = int(sys.argv[5]);

flow_dict = get_flowdict(fully_connected_layers, planar_layers, spinner_layers, nonlin_spinner_layers);
flow_ids = flow_dict['flow_ids'];
flow_repeats = flow_dict['flow_repeats'];
print_flowdict(flow_dict);

cost_type = 'KL';
K_eta = 10;
M_eta = 1000;
stochastic_eta = False;
L = 8;
batch_norm = False;
dropout = False;
lr_order = -3;
random_seed = 0;
max_iters = 10000;
check_rate = 100;

X, train_KLs, it = train_efn(exp_fam, D, flow_dict, cost_type, K_eta, M_eta, stochastic_eta, \
	                       L, batch_norm, dropout, lr_order, random_seed, max_iters, check_rate);
