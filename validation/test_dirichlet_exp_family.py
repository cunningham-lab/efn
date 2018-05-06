from train_efn import train_efn
import numpy as np
from matplotlib import pyplot as plt
import os, sys
from efn_util import get_flowdict, print_flowdict

os.chdir('../');

exp_fam = 'dirichlet';
D = int(sys.argv[1]);
fully_connected_layers = int(sys.argv[2]);
planar_layers = int(sys.argv[3]);
spinner_layers = int(sys.argv[4]);
nonlin_spinner_layers = int(sys.argv[5]);
L = int(sys.argv[6]);
upl_tau = float(sys.argv[7]);

cost_type = 'KL';
ncons = D;
K_eta = 100
M_eta = 1000;
stochastic_eta = True;
batch_norm = False;
dropout = False;
lr_order = -3;
random_seed = 0;
max_iters = 20000;
check_rate = 100;

flow_dict = get_flowdict(fully_connected_layers, planar_layers, spinner_layers, nonlin_spinner_layers);
print_flowdict(flow_dict);

X, KLs, it = train_efn(exp_fam, D, flow_dict, cost_type, K_eta, M_eta, \
                       stochastic_eta, L, upl_tau, batch_norm, dropout, lr_order, random_seed, max_iters, check_rate);
