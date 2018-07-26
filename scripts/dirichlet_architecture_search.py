from train_nf import train_nf
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import os
import sys
from efn_util import drawEtas, get_flowdict, print_flowdict, get_flowstring

os.chdir('../');

exp_fam = 'dirichlet';
D = int(sys.argv[1]);
planar_layers = int(sys.argv[3]);
spinner_layers = int(sys.argv[4]);
nonlin_spinner_layers = int(sys.argv[5]);

flow_dict = get_flowdict(fully_connected_layers, planar_layers, spinner_layers, nonlin_spinner_layers);
flow_ids = flow_dict['flow_ids'];
flow_repeats = flow_dict['flow_repeats'];
print_flowdict(flow_dict);
flowstring = get_flowstring(flow_dict);


cost_type = 'KL';
M_eta = 1000;
lr_order = -2;
random_seed = 0;
check_rate = 100;
max_iters = 10000;


fname = 'dirichlet_architecture_%s.npz' % flowstring;
print(fname);

np.random.seed(0);
etas, params = drawEtas(exp_fam, D, 1);
log_P, X, R2s, KLs, it = train_nf(exp_fam, params, flow_dict, cost_type, M_eta, lr_order, random_seed, max_iters, check_rate);
print('saving', fname);
np.savez(fname, R2s=R2s, KLs=KLs, X=X, logP=log_P, check_rate=check_rate);
print('done');


