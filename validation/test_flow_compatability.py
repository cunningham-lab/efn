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
dist_seed = int(sys.argv[3]);

TIF_flow_type, nlayers, lr_order = model_opt_hps(exp_fam, D);
lr_order = -3;
nlayers = 30;
flow_dict = {'latent_dynamics':None, \
             'TIF_flow_type':TIF_flow_type, \
             'repeats':nlayers};

T = 1;
cost_type = 'KL';
M_eta = 1000;
give_inverse_hint = False;
random_seed = 0;
check_rate = 100;
min_iters = 10000;
max_iters = 20000;

fam_class = family_from_str(exp_fam);
family = fam_class(D);

np.random.seed(dist_seed);
_, _, _, params = family.draw_etas(1, 'eta', give_inverse_hint);
params = params[0];
params.update({'dist_seed':dist_seed});

log_P, X, R2s, KLs, it = train_nf(family, params, flow_dict, cost_type, M_eta, lr_order, \
	                              random_seed, min_iters, max_iters, check_rate);


