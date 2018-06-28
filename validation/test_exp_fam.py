from train_efn import train_efn
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
import os, sys
from efn_util import get_flowdict, print_flowdict
from families import family_from_str

os.chdir('../');

exp_fam = str(sys.argv[1]);
D = int(sys.argv[2]);
linear_layers = int(sys.argv[3]);
planar_layers = int(sys.argv[4]);
spinner_layers = int(sys.argv[5]);
nonlin_spinner_layers = int(sys.argv[6]);
give_hint = not(int(sys.argv[7]) == 0);
param_net_input_type = str(sys.argv[8]);
random_seed = int(sys.argv[9]);

flow_dict = get_flowdict(0, linear_layers, 0, planar_layers, spinner_layers, nonlin_spinner_layers);
print_flowdict(flow_dict);

cost_type = 'KL';
K_eta = 100;
M_eta = 100;
stochastic_eta = True;
L = 4;
upl_tau = 0.5;

lr_order = -3;
max_iters = 20000;
check_rate = 100;

fam_class = family_from_str(exp_fam);
family = fam_class(D);

X, train_KLs, it = train_efn(family, flow_dict, param_net_input_type, cost_type, K_eta, M_eta, stochastic_eta, \
	                         give_hint, lr_order, random_seed, max_iters, check_rate);
