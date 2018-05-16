from train_nf import train_nf
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import os, sys
from efn_util import get_flowdict, print_flowdict, drawEtas

os.chdir('../');

exp_fam = str(sys.argv[1]);
D = int(sys.argv[2]);
give_inverse_hint = int(sys.argv[3]) == 1;
random_seed = int(sys.argv[4]);

if (exp_fam == 'inv_wishart'):
	sqrtD = int(np.sqrt(D));
	planar_layers = int(sqrtD*(sqrtD+1)/2);
else:
	planar_layers = D;
if (planar_layers < 20):
	planar_layers = 20;

flow_dict = get_flowdict(0, planar_layers, 0, 0);
flow_ids = flow_dict['flow_ids'];
flow_repeats = flow_dict['flow_repeats'];
print_flowdict(flow_dict);

cost_type = 'KL';
M_eta = 1000;
stochastic_eta = True;
lr_order = -4;
max_iters = 50000;
check_rate = 100;

if (exp_fam == 'dir_dir'):
	model_info = {'Ndrawtype':'1','subclass':'NF1', 'extrastr':''}
else:
	model_info = {'subclass':NF1, 'extrastr':''};

np.random.seed(random_seed);
eta, param_net_input, Tx_input, params = drawEtas(exp_fam, D, 1, model_info, give_inverse_hint);

log_p_zs, X, train_R2s, train_KLs, it = train_nf(exp_fam, params, flow_dict, cost_type, M_eta, \
	                         					 model_info, lr_order, random_seed, max_iters, check_rate);
