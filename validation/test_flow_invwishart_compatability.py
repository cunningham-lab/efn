from train_nf import train_nf
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import os, sys
from plot_utils import plotMefnTraining
from efn_util import get_flowdict, print_flowdict

os.chdir('../');

exp_fam = 'inv_wishart';
D = int(sys.argv[1]);
fully_connected_layers = int(sys.argv[2]);
planar_layers = int(sys.argv[3]);
spinner_layers = int(sys.argv[4]);
nonlin_spinner_layers = int(sys.argv[5]);

flow_dict = get_flowdict(fully_connected_layers, planar_layers, spinner_layers, nonlin_spinner_layers);
print_flowdict(flow_dict);

cost_type = 'KL';
M_eta = 1000;
lr_order = -2;
random_seed = 0;

ms = [D*2];
Psis = [];
Psi = np.eye(D);
Psi[0,0] = 2.0;
Psis.append(Psi);

max_iters = 100000;
check_rate = 100;
num_Psis = len(Psis);
for i in range(num_Psis):
	#alpha = alphas[i];
	Psi = np.expand_dims(Psis[i], 0);
	m = np.array([ms[i]]);
	D = Psi.shape[1]**2;
	params = {'Psi': Psi, 'm':m, 'D':D};
	log_P, X, R2s, KLs, it = train_nf(exp_fam, params, flow_dict, cost_type, M_eta, \
		                              lr_order, random_seed, max_iters, check_rate);
	print('xshape', X.shape);
	print('took %d iterations' % it)
	#plotMefnTraining(exp_fam, R2s, KLs, X, log_P, params, check_rate, it, 'dirichlet %d' % (i+1));
plt.show();


