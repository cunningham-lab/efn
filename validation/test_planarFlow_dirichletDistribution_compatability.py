from train_mefn import train_mefn
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import os

os.chdir('../');

exp_fam = 'dirichlet';
D = 3;
flow_id = 'linear1';
cost_type = 'KL';
M = 300;
lr_order = -3;
random_seed = 0;

alpha = [1,1,1];
params = {'alpha': alpha, 'D':D};
X, R2s, it = train_mefn(exp_fam, params, flow_id, cost_type, M, lr_order, random_seed);
