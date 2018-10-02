from train_nf import train_nf
import numpy as np
import os
import sys
from efn_util import model_opt_hps
from tf_util.families import dirichlet, multivariate_normal, inv_wishart, hierarchical_dirichlet, \
                     dirichlet_multinomial, truncated_normal_poisson, family_from_str

os.chdir('../');

exp_fam = str(sys.argv[1]);
D = int(sys.argv[2]);
monkey = int(sys.argv[3]);
neuron = int(sys.argv[4]);
ori = int(sys.argv[5]);

profile = True;

dir_str = exp_fam;

resp_info = {'monkey':monkey, \
			 'neuron':neuron, \
             'ori':ori};

TIF_flow_type, nlayers, scale_layer, lr_order = model_opt_hps(exp_fam, D);
lr_order = -3;
nlayers = 30;
flow_dict = {'latent_dynamics':None, \
			 'scale_layer':scale_layer, \
             'TIF_flow_type':TIF_flow_type, \
             'repeats':nlayers};

cost_type = 'KL';
M_eta = 200;
random_seed = 0;
min_iters = 25;
max_iters = 25;
check_rate = 100;

fam_class = family_from_str(exp_fam);
family = fam_class(D);

family.load_data();
train, test = family.select_train_test_sets(0);

_, _, _, params = family.draw_etas(1, 'eta', False, True, resp_info);
params = params[0];
print(params);
params.update({'dist_seed':params['data_ind']});

log_P, X, R2s, KLs, it = train_nf(family, params, flow_dict, cost_type, M_eta, lr_order, \
	                              random_seed, min_iters, max_iters, check_rate, dir_str, profile);


