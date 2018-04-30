from train_efn import train_efn
import numpy as np
from matplotlib import pyplot as plt
import os, sys

os.chdir('../');

exp_fam = 'inv_wishart';
D = int(sys.argv[1]);
flow_id = str(sys.argv[2]);
cost_type = 'KL';
ncons = D+1;
L_theta = int(sys.argv[3]);
upl_fac = int(sys.argv[4]);
upl_theta = upl_fac*ncons;
stochastic_eta = False;
lr_order = -2;
theta_nn_hps = {'L':L_theta, 'upl':upl_theta};
max_iters = 10000;
check_rate = 100;

num_checks = (max_iters // check_rate) + 1;
Ks = [10, 5, 10];
num_Ks = len(Ks);
Ms = [1000, 100, 1000];
num_Ms = len(Ms);
N = 20;

KLs = np.zeros((num_Ks, num_Ms, N, num_checks));
R2s = np.zeros((num_Ks, num_Ms, N, num_checks));

for i in range(num_Ks):
	K_eta = Ks[i];
	for j in range(num_Ms):
		M_eta = Ms[j];
		N_k = N // K_eta;
		for n in range(0,N_k):
			random_seed = n;
			print('K_eta=%d, M_eta=%d, seed=%d/%d' % (K_eta, M_eta, random_seed+1, N_k));
			X, trainR2s, trainKLs, it = train_efn(exp_fam, D, flow_id, cost_type, K_eta, M_eta, \
						                     stochastic_eta, theta_nn_hps, lr_order, \
						                     random_seed, max_iters, check_rate);
			KLs[i, j, (n*K_eta):((n+1)*K_eta)] = trainKLs.T;
			R2s[i, j, (n*K_eta):((n+1)*K_eta)] = trainR2s.T;
			np.savez('M_K_%s_lyrs=%d_upl=%d_tradeoff.npz' % (flow_id, L_theta, upl_theta), KLs=KLs, R2s=R2s, \
                      check_rate=check_rate, Ks=Ks, Ms=Ms, N=N);


