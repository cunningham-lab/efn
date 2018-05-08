from train_efn import train_efn
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import os, sys
from efn_util import get_flowdict, print_flowdict

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

flow_dict = get_flowdict(1, 0, 0, 0);
flow_ids = flow_dict['flow_ids'];
flow_repeats = flow_dict['flow_repeats'];
print_flowdict(flow_dict);

cost_type = 'KL';
K_eta = 100;
M_eta = 1000;
stochastic_eta = True;
lr_order = -3;
max_iters = 50000;
check_rate = 100;

X, train_KLs, it = train_efn(exp_fam, D, flow_dict, cost_type, K_eta, M_eta, stochastic_eta, \
	                         give_inverse_hint, lr_order, random_seed, max_iters, check_rate);
