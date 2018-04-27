from train_efn import train_efn
import numpy as np
from matplotlib import pyplot as plt
import os, sys

os.chdir('../');

exp_fam = 'dirichlet';
D = int(sys.argv[1]);
flow_id = 'planar2';
cost_type = 'KL';
ncons = D;
L_theta = int(sys.argv[2]);
upl_fac = int(sys.argv[3]);
upl_theta = upl_fac*ncons;
K_eta = 1;
M_eta = 1000;
stochastic_eta = False;
lr_order = -3;
random_seed = int(sys.argv[4]);
theta_nn_hps = {'L':L_theta, 'upl':upl_theta};
max_iters = 10000;
check_rate = 1000;

X, KLs, it = train_efn(exp_fam, D, flow_id, cost_type, K_eta, M_eta, \
                       stochastic_eta, theta_nn_hps, lr_order, random_seed, max_iters, check_rate);
