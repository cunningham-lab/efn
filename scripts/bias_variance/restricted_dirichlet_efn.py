from train_efn import train_efn
import numpy as np
from scipy.stats import multivariate_normal
from tf_util.families import family_from_str
from efn_util import model_opt_hps
import os, sys

os.chdir('../../');

scale = float(sys.argv[1]);
random_seed = int(sys.argv[2]);

exp_fam = 'dirichlet';
D = 3;
nlayers = 2
give_inverse_hint = False
dir_str = 'dirichlet_bv';

TIF_flow_type, _, lr_order = model_opt_hps(exp_fam, D);

flow_dict = {'latent_dynamics':None, \
             'TIF_flow_type':TIF_flow_type, \
             'repeats':nlayers};

mu = np.array([1.0, 2.0, 3.0]);
eta_dist = {'family':'isotropic_truncated_normal', 'mu':mu, \
            'scale':scale};

fam_class = family_from_str(exp_fam);
family = fam_class(D, 1, eta_dist);

param_net_input_type = 'eta';
cost_type = 'KL';
K_eta = 100;
M_eta = 1000;
stochastic_eta = True;
dist_seed = 0;
min_iters = 50000;
max_iters = 50000;
check_rate = 100;

X, train_KLs, it = train_efn(family, flow_dict, param_net_input_type, cost_type, K_eta, M_eta, \
	                         stochastic_eta, give_inverse_hint, lr_order, dist_seed, random_seed, \
	                         min_iters, max_iters, check_rate, dir_str);
