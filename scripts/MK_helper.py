from train_efn import train_efn
import numpy as np 
from matplotlib import pyplot as plt
import os, sys

os.chdir('../');

exp_fam = str(sys.argv[1]);
D = int(sys.argv[2]);
K_eta = int(sys.argv[3]);
M_eta = int(sys.argv[4]);
flow_id = str(sys.argv[5]);
cost_type = 'KL';
ncons = D;
L_theta = int(sys.argv[6]);
upl_fac = int(sys.argv[7]);
upl_theta = upl_fac*ncons;
stochastic_eta = False;
lr_order = -3;
theta_nn_hps = {'L':L_theta, 'upl':upl_theta};
max_iters = 50000;
check_rate = 200;
random_seed = int(sys.argv[8]);

X, trainKLs, it = train_efn(exp_fam, D, flow_id, cost_type, K_eta, M_eta, \
			                                 stochastic_eta, theta_nn_hps, lr_order, \
						                     random_seed, max_iters, check_rate);


