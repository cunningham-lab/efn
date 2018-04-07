from train_network import train_network
import numpy as np
from matplotlib import pyplot as plt
from efn_util import load_constraint_info
from scipy.stats import multivariate_normal
import os

os.chdir('../');

flow_id = 'planar4';
cost_type = 'KL';
L = 1;
n = 100;
K_eta = None;
stochastic_eta = False;
param_network = False;
lr_order = -3;
random_seed = 1;

constraint_id = 'dirichlet_1';
D_Z, K_eta_params, params, constraint_type = load_constraint_info(constraint_id);
X, R2s, it = train_network(constraint_id, flow_id, cost_type,  L, n, K_eta, \
                       stochastic_eta, param_network, lr_order, random_seed);
