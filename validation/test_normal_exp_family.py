from train_network import train_network
import numpy as np
from matplotlib import pyplot as plt
from efn_util import load_constraint_info
from scipy.stats import multivariate_normal
import os, sys

os.chdir('../');

constraint_id = 'normal';
flow_id = 'linear1';
D = int(sys.argv[1]);
ncons = D+D**2;
cost_type = 'KL';
L = int(sys.argv[2]);
upl_fac = int(sys.argv[3]);
units_per_layer = upl_fac*ncons;
n = 1000;
K_eta = 25;
stochastic_eta = True;
single_dist = False;
lr_order = -3;
random_seed = 0;
X, R2s, it = train_network(constraint_id, D, flow_id, cost_type,  L, \
	                       units_per_layer, n, K_eta, \
                           stochastic_eta, single_dist, lr_order, random_seed);