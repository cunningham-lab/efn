from train_nf_new import train_nf
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
from plot_utils import plotMefnTraining
from efn_util import get_flowdict, print_flowdict
from families import dirichlet, multivariate_normal, inv_wishart, dirichlet_dirichlet

os.chdir('../');

exp_fam = str(sys.argv[1]);
D = int(sys.argv[2]);
fully_connected_layers = int(sys.argv[3]);
planar_layers = int(sys.argv[4]);
spinner_layers = int(sys.argv[5]);
nonlin_spinner_layers = int(sys.argv[6]);

flow_dict = get_flowdict(fully_connected_layers, planar_layers, spinner_layers, nonlin_spinner_layers);
print_flowdict(flow_dict);

T = 1;
cost_type = 'KL';
M_eta = 1000;
give_inverse_hint = False;
lr_order = -3;
random_seed = 0;
check_rate = 100;
max_iters = 20000;
ndraws = 1;

model_info = {'subclass':'NF1', 'extrastr':''};

if (exp_fam == 'dirichlet'):
	family = dirichlet(D);
elif (exp_fam == 'normal'):
	family = multivariate_normal(D);
elif (exp_fam == 'inv_wishart'):
	family = inv_wishart(D);
elif (exp_fam == 'dirichlet_dirichlet'):
	family = dirichlet_dirichlet(D);

np.random.seed(0);
_, _, _, params = family.draw_etas(ndraws, 'eta', give_inverse_hint);
for i in range(ndraws):
	params_i = params[i];
	log_P, X, R2s, KLs, it = train_nf(family, exp_fam, params_i, flow_dict, cost_type, \
		                              M_eta, model_info, lr_order, \
		                              random_seed, max_iters, check_rate);


