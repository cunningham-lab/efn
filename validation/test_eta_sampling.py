from train_network import train_network
from matplotlib import pyplot as plt
from efn_util import factors
import os
import numpy as np
os.chdir('../');

constraint_id = 'normal_2VL'
flow_id = 'linear1';
cost_type = 'KL';
L = 1;
n = 300;
K_eta = None;
stochastic_eta = False;
param_network = True;
lr_order = -3;

nruns = 5;
training_lattice_its = np.zeros((nruns,));
for i in range(nruns):
	X, R2s, it = train_network(constraint_id, flow_id, cost_type, L, n, K_eta,\
                       			stochastic_eta, param_network, lr_order, i);
	training_lattice_its[i] = it;
	print(training_lattice_its);

stochastic_eta = True;
K_etas = factors(n);
num_K_etas = len(K_etas);
stochastic_eta_its = np.zeros((num_K_etas, nruns));
for i in range(num_K_etas):
	K_eta = K_etas[i];
	for j in range(nruns):
		print('K_eta %d, run %d' % (K_eta, j+1));
		X, R2s, it = train_network(constraint_id, flow_id, cost_type, L, n, K_eta,\
	                       			stochastic_eta, param_network, lr_order, j);
		stochastic_eta_its[i,j] = it;
		print(stochastic_eta_its);


np.savez('2VL_results.npz', tl_its=training_lattice_its, K_etas=K_etas, se_its=stochastic_eta_its);