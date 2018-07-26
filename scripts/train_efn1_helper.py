from train_efn import train_efn
import numpy as np
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

K = 1;
M = 1000;
param_net_input_type = 'eta';
cost_type = 'KL'
stochastic_eta = False;
min_iters = 100000;
max_iters = 1000000;
check_rate = 100;

X, train_KLs, it = train_efn(family, flow_dict, param_net_input_type, cost_type, K, M, \
	                         stochastic_eta, give_inverse_hint, lr_order, dist_seed, random_seed, \
	                         min_iters, max_iters, check_rate, dir_str);
