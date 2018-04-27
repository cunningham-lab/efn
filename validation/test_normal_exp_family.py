from train_efn import train_efn
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import os, sys

os.chdir('../');

exp_fam = 'normal';
D = int(sys.argv[1]);
flow_id = 'linear1';
cost_type = 'KL';
K_eta = 10;
M_eta = 1000;
L = int(sys.argv[2]);
upl_fac = int(sys.argv[3]);
ncons = D+D**2;
upl = upl_fac*ncons;
theta_nn_hps = {'L':L, 'upl':upl};
stochastic_eta = False;
lr_order = -3;
random_seed = 0;
max_iters = 10000;
check_rate = 1;

X, train_KLs, it = train_efn(exp_fam, D, flow_id, cost_type, K_eta, M_eta, stochastic_eta, \
	                       theta_nn_hps, lr_order, random_seed, max_iters, check_rate);
