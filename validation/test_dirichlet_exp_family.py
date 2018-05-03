from train_efn import train_efn
import numpy as np
from matplotlib import pyplot as plt
import os, sys

os.chdir('../');

exp_fam = 'dirichlet';
D = int(sys.argv[1]);
flow_id = 'planar20';
cost_type = 'KL';
ncons = D;
K_eta = int(sys.argv[2]);
M_eta = 110;
stochastic_eta = True;
batch_norm = False;
dropout = False;
lr_order = -2;
random_seed = 0;
max_iters = 10000;
check_rate = 100;

L_theta = 8;


X, KLs, it = train_efn(exp_fam, D, flow_id, cost_type, K_eta, M_eta, \
                       stochastic_eta, L_theta, batch_norm, dropout, lr_order, random_seed, max_iters, check_rate);
