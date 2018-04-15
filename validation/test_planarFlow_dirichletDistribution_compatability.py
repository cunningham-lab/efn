from train_network import train_network
import numpy as np
from matplotlib import pyplot as plt
from efn_util import load_constraint_info
from scipy.stats import multivariate_normal
import os

os.chdir('../');

constraint_id = 'dirichlet_1';
D_Z = 2;
flow_id = 'planar4';
cost_type = 'KL';
L = 1;
units_per_layer = 1;
M = 100;
K_eta = 1;
stochastic_eta = False;
single_dist = True;
lr_order = -3;
random_seed = 0;

D_Z, K_eta_params, params, constraint_type = load_constraint_info(constraint_id);
X, R2s, it = train_network(constraint_id, D_Z, flow_id, cost_type,  L, units_per_layer, M, K_eta, \
                       stochastic_eta, single_dist, lr_order, random_seed);
