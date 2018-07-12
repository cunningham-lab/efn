from train_efn import train_efn
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import os, sys
from families import family_from_str

os.chdir('../');

exp_fam = str(sys.argv[1]);
D = int(sys.argv[2]);
flow_type_str = str(sys.argv[3]);
nlayers = int(sys.argv[4]);
give_inverse_hint = int(sys.argv[5]) == 1;
random_seed = int(sys.argv[6]);

if (flow_type_str == 'P'):
	TIF_flow_type = 'PlanarFlowLayer';
elif (flow_type_str == 'A'):
	TIF_flow_type = 'AffineFlowLayer';

flow_dict = {'latent_dynamics':None, \
             'TIF_flow_type':TIF_flow_type, \
             'repeats':nlayers};

fam_class = family_from_str(exp_fam);
family = fam_class(D);

cost_type = 'KL';
param_net_input_type = 'eta';
K_eta = 100;
M_eta = 100;
stochastic_eta = True;
lr_order = -3;
max_iters = 1000000;
check_rate = 200;

X, train_KLs, it = train_efn(family, flow_dict, param_net_input_type, cost_type, K_eta, M_eta, \
	                         stochastic_eta, give_inverse_hint, lr_order, random_seed, max_iters, check_rate);
