from train_efn import train_efn
import numpy as np 
from matplotlib import pyplot as plt
import os, sys
from efn_util import get_flowdict

os.chdir('../');

exp_fam = str(sys.argv[1]);
D = int(sys.argv[2]);
K_eta = int(sys.argv[3]);
M_eta = int(sys.argv[4]);
cost_type = 'KL';
ncons = D;
L = 8;
upl_tau = 0.5;
batch_norm = False;
dropout = False;
stochastic_eta = False;
lr_order = -3;
max_iters = 50000;
check_rate = 200;
random_seed = int(sys.argv[5]);

# setup the normalizing flow
fully_connected_layers = 0;
planar_layers = 40;
spinner_layers = 0;
nonlin_spinner_layers = 0;
flow_dict = get_flowdict(fully_connected_layers, planar_layers, spinner_layers, nonlin_spinner_layers);


X, trainKLs, it = train_efn(exp_fam, D, flow_dict, cost_type, K_eta, M_eta, \
			                                 stochastic_eta, L, upl_tau, batch_norm, dropout, lr_order, \
						                     random_seed, max_iters, check_rate);


