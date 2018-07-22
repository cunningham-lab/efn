from train_nf import train_nf
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import os, sys
from families import family_from_str
from efn_util import model_opt_hps

os.chdir('../');

exp_fam = str(sys.argv[1]);
D = int(sys.argv[2]);
give_inverse_hint = int(sys.argv[3]) == 1;
dist_seed = int(sys.argv[4]);
random_seed = int(sys.argv[5]);
dir_str = str(sys.argv[6]);

TIF_flow_type, nlayers, lr_order = model_opt_hps(exp_fam, D);

flow_dict = {'latent_dynamics':None, \
             'TIF_flow_type':TIF_flow_type, \
             'repeats':nlayers};

fam_class = family_from_str(exp_fam);
family = fam_class(D);

cost_type = 'KL';
M_eta = 1000;
max_iters = 1000000;
check_rate = 100;

param_net_input_type = 'eta';  # I should generalize this draw_etas function to accept a None
np.random.seed(dist_seed);
eta, param_net_input, Tx_input, params = family.draw_etas(1, param_net_input_type, give_inverse_hint);
params = params[0];
params.update({'dist_seed':dist_seed});

log_p_zs, X, train_R2s, train_KLs, it = train_nf(family, params, flow_dict, cost_type, M_eta, lr_order, \
	                         					 random_seed, max_iters, check_rate, dir_str);
