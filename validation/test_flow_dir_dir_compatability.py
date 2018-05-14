from train_nf import train_nf
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal, dirichlet
import os, sys
from efn_util import get_flowdict, print_flowdict, drawEtas
os.chdir('../');

exp_fam = 'dir_dir';
D = 3;

planar_layers = int(sys.argv[1]);
fully_connected_layers = 0;
spinner_layers = 0;
nonlin_spinner_layers = 0;

flow_dict = get_flowdict(fully_connected_layers, planar_layers, spinner_layers, nonlin_spinner_layers);
flow_ids = flow_dict['flow_ids'];
flow_repeats = flow_dict['flow_repeats'];
print_flowdict(flow_dict);

cost_type = 'KL';
M_eta = 100;
lr_order = -3;
random_seed = 0;
max_iters = 8000;
check_rate = 100;

alpha_0 = np.array([1.0,5.0,10.0]);
np.random.seed(0);
beta = 1.0;
N = 100;
dist1 = dirichlet(alpha_0);
z = np.array([[.98, .01, .01]]);
betaz = beta*z[0];
dist2 = dirichlet(beta*z[0]);
x = dist2.rvs(N).T;

alpha_0s = np.array([alpha_0]);
betas = np.array([beta]);
Ns = np.array([N]);
zs = np.array(z);
xs = np.array([x]);

params = {'alpha_0s':alpha_0s, 'betas':betas, 'xs':xs, 'zs':zs, 'Ns':Ns, 'D':D};

log_P, X, R2s, KLs, it = train_nf(exp_fam, params, flow_dict, cost_type, \
		                          M_eta, lr_order, random_seed, max_iters, check_rate);
