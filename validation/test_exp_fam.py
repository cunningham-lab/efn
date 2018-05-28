from train_efn import train_efn
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
import os, sys
from efn_util import get_flowdict, print_flowdict
from families import dirichlet, multivariate_normal, inv_wishart, hierarchical_dirichlet, \
                     dirichlet_multinomial, truncated_normal_poisson

os.chdir('../');

exp_fam = str(sys.argv[1]);
D = int(sys.argv[2]);
fully_connected_layers = int(sys.argv[3]);
planar_layers = int(sys.argv[4]);
spinner_layers = int(sys.argv[5]);
nonlin_spinner_layers = int(sys.argv[6]);
param_net_input_type = str(sys.argv[7]);

flow_dict = get_flowdict(fully_connected_layers, planar_layers, spinner_layers, nonlin_spinner_layers);
print_flowdict(flow_dict);

cost_type = 'KL';
K_eta = 100;
M_eta = 100;
stochastic_eta = True;
L = 4;
upl_tau = 0.5;
give_hint = False;
lr_order = -3;
random_seed = 0;
max_iters = 10000;
check_rate = 100;

if (exp_fam == 'dirichlet'):
	family = dirichlet(D);
elif (exp_fam == 'normal'):
	family = multivariate_normal(D);
elif (exp_fam == 'inv_wishart'):
	family = inv_wishart(D);
elif (exp_fam == 'hierarchical_dirichlet'):
	family = hierarchical_dirichlet(D);
elif (exp_fam == 'dirichlet_multinomial'):
	family = dirichlet_multinomial(D);
elif (exp_fam == 'truncated_normal_poisson'):
	family = truncated_normal_poisson(D);

X, train_KLs, it = train_efn(family, flow_dict, param_net_input_type, cost_type, K_eta, M_eta, stochastic_eta, \
	                         give_hint, lr_order, random_seed, max_iters, check_rate);
