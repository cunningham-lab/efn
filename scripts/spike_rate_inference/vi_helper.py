from train_vi import train_vi
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import os, sys
from efn_util import get_flowdict, print_flowdict
import scipy.io as sio

os.chdir('../');

exp_fam = 'prp_tn';
R = int(sys.argv[1]);
N = int(sys.argv[2]);
random_seed = int(sys.argv[3]);
D = 20;
planar_layers = 30;
random_seed = 0;

if (R > N):
	print('R > N, so quitting');
	exit();

flow_dict = get_flowdict(0, planar_layers, 0, 0);
flow_ids = flow_dict['flow_ids'];
flow_repeats = flow_dict['flow_repeats'];
print_flowdict(flow_dict);

datapath = '/Users/sbittner/Documents/efn/data/spike_counts_neuron4.mat';
data = sio.loadmat(datapath);
datax = data['x'];
extrastr = 'R=%d_N=%d_' % (R,N);
model_info = {'subclass':'VI', 'extrastr':extrastr, 'R':R, 'trainx':datax[:N,:], 'testx':datax[100:,:]};

cost_type = 'KL';
M = 1000;
lr_order = -3;
max_iters = 50000;
check_rate = 100;

log_p_zs, X, train_R2s, train_KLs, it = train_vi(exp_fam, D, flow_dict, cost_type, M, model_info, \
		                         					 lr_order, random_seed, max_iters, check_rate);
