from train_nf import train_nf
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
from plot_utils import plotMefnTraining
from efn_util import model_opt_hps
from families import dirichlet, multivariate_normal, inv_wishart, hierarchical_dirichlet, \
                     dirichlet_multinomial, truncated_normal_poisson, family_from_str

os.chdir('../');


exp_fam = str(sys.argv[1]);
D = int(sys.argv[2]);
monkey = int(sys.argv[3]);
neuron = int(sys.argv[4]);
ori = int(sys.argv[5]);

resp_info = {'monkey':monkey, \
			 'neuron':neuron, \
             'ori':ori};

dir_str = exp_fam;

TIF_flow_type = 'PlanarFlowLayer';
lr_order = -3;
nlayers = 20;
flow_dict = {'latent_dynamics':None, \
             'TIF_flow_type':TIF_flow_type, \
             'repeats':nlayers};

T = 1;
cost_type = 'KL';
M_eta = 1000;
give_inverse_hint = True;
random_seed = 0;
check_rate = 100;
max_iters = 10000;

fam_class = family_from_str(exp_fam);
family = fam_class(D);

family.load_data();
train, test = family.select_train_test_sets(0);

dist_seed = family.resp_info_to_ind(resp_info);
_, _, _, params = family.draw_etas(1, 'eta', give_inverse_hint, True, resp_info);
params = params[0];
print(params);
params.update({'dist_seed':dist_seed});

log_P, X, R2s, KLs, it = train_nf(family, params, flow_dict, cost_type, M_eta, lr_order, \
	                              random_seed, max_iters, check_rate, dir_str);


