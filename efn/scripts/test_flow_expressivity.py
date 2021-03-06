from train_nf import train_nf
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import os
import sys
from efn_util import drawEtas, get_flowdict, print_flowdict, get_flowstring

os.chdir('../');

exp_fam = str(sys.argv[1]);
fully_connected_layers = int(sys.argv[2]);
planar_layers = int(sys.argv[3]);
spinner_layers = int(sys.argv[4]);
nonlin_spinner_layers = int(sys.argv[5]);

flow_dict = get_flowdict(fully_connected_layers, planar_layers, spinner_layers, nonlin_spinner_layers);
flow_ids = flow_dict['flow_ids'];
flow_repeats = flow_dict['flow_repeats'];
print_flowdict(flow_dict);
flowstring = get_flowstring(flow_dict);

Ds = [2, 4, 8, 16];
num_Ds = len(Ds);
cost_type = 'KL';
M_eta = 1000;
lr_order = -3;
random_seed = 0;
check_rate = 200;
max_iters = 20000;
ndraws = 10;

R2s = np.zeros((num_Ds, ndraws));
KLs = np.zeros((num_Ds, ndraws));

fname = 'flow_expressivity_test_%s_%s.npz' % (exp_fam, flowstring);
print(fname);
for i in range(num_Ds):
	D = Ds[i];
	for j in range(ndraws):
		print('D=%d, draw=%d' % (D, j+1));
		np.random.seed(j);
		etas, param_net_inputs, params = drawEtas(exp_fam, D, 1, False);
		log_P, X, R2s_ij, KLs_ij, it = train_nf(exp_fam, params, flow_dict, cost_type, M_eta, lr_order, random_seed, max_iters, check_rate);
		R2s[i,j] = R2s_ij[-1,0];
		KLs[i,j] = KLs_ij[-1,0];
	print('saving', fname);
	np.savez(fname, R2s=R2s, KLs=KLs, Ds=Ds);
print('done');


