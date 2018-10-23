from train_nf import train_nf
import numpy as np
from scipy.stats import multivariate_normal
from tf_util.families import family_from_str
from efn_util import model_opt_hps
import os, sys

os.chdir('../../');

random_seed = int(sys.argv[1]);

exp_fam = 'dirichlet';
D = 3;
nlayers = 2;
give_inverse_hint = False;
dist_seed = 0;
dir_str = 'dirichlet_bv';

TIF_flow_type, _, lr_order = model_opt_hps(exp_fam, D);

flow_dict = {'latent_dynamics':None, \
             'TIF_flow_type':TIF_flow_type, \
             'repeats':nlayers};

fam_class = family_from_str(exp_fam);
family = fam_class(D);

cost_type = 'KL';
M_eta = 1000;
min_iters = 50000;
max_iters = 50000;
check_rate = 100;

params = {'alpha':np.array([1.0, 2.0, 3.0]), 'dist_seed':dist_seed};

log_p_zs, X, train_R2s, train_KLs, it = train_nf(family, params, flow_dict, cost_type, M_eta, lr_order, \
	                         					 random_seed, min_iters, max_iters, check_rate, dir_str);
