from train_efn import train_efn
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
subclass = str(sys.argv[5]);
# 'EFN1' - orig class
# 'EFN1a' - just prior fed into param network
# 'EFN1b' - just likelihood params fed into param network
# 'EFN1c' - just data fed into param network
# 'NF'


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

if (exp_fam == 'dir_dir'):
	model_info = {'Ndrawtype':'1', 'subclass':subclass, 'extrastr':'1samp_'};
else:
	model_info = {'subclass':subclass, 'extrastr':'1samp_'};

cost_type = 'KL';
K_eta = 100;
M_eta = 100;
stochastic_eta = False;
lr_order = -4;
max_iters = 20000;
check_rate = 100;

if (subclass[:3] == 'EFN'):
	X, train_KLs, it = train_efn(exp_fam, D, flow_dict, cost_type, K_eta, M_eta, model_info, stochastic_eta, \
	                         give_inverse_hint, lr_order, random_seed, max_iters, check_rate);
elif (subclass == 'NF1'):
	np.random.seed(random_seed);
	eta, param_net_input, Tx_input, params = drawEtas(exp_fam, D, 1, model_info, give_inverse_hint);

	log_p_zs, X, train_R2s, train_KLs, it = train_nf(exp_fam, params, flow_dict, cost_type, M_eta, model_info, \
		                         					 lr_order, random_seed, max_iters, check_rate);
