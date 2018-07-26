from train_nf import train_nf
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
from efn_util import model_opt_hps
from families import dirichlet, multivariate_normal, inv_wishart, hierarchical_dirichlet, \
                     dirichlet_multinomial, truncated_normal_poisson, family_from_str

os.chdir('../');


exp_fam = str(sys.argv[1]);
D = int(sys.argv[2]);
give_inverse_hint = int(sys.argv[3]) == 1;
dist_seed = int(sys.argv[4]);
dir_str = str(sys.argv[5]);
#monkey = int(sys.argv[3]);
#neuron = int(sys.argv[4]);
#ori = int(sys.argv[5]);

#resp_info = {'monkey':monkey, \
#			 'neuron':neuron, \
#             'ori':ori};

TIF_flow_type, nlayers, lr_order = model_opt_hps(exp_fam, D);

flow_dict = {'latent_dynamics':None, \
             'TIF_flow_type':TIF_flow_type, \
             'repeats':nlayers};

T = 1;
prior = {'N':1};

cost_type = 'KL';
M_eta = 100;
random_seed = 0;
check_rate = 100;
min_iters = 100000;
max_iters = 1000000;

fam_class = family_from_str(exp_fam);
family = fam_class(D, T, prior);

family.load_data();
train, test = family.select_train_test_sets(0);

np.random.seed(dist_seed);
_, _, _, params = family.draw_etas(1, 'eta', give_inverse_hint);
params = params[0];
print(params);
params.update({'dist_seed':dist_seed});

log_P, X, R2s, KLs, it = train_nf(family, params, flow_dict, cost_type, M_eta, lr_order, \
	                              random_seed, min_iters, max_iters, check_rate, dir_str);


