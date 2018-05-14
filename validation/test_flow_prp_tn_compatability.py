from train_nf import train_nf
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import os, sys
from efn_util import get_flowdict, print_flowdict, drawEtas, drawPoissonRates, drawPoissonCounts
from plot_utils import plotContourTruncatedNormal
os.chdir('../');

exp_fam = 'prp_tn';
D = int(sys.argv[1]);
planar_layers = int(sys.argv[2]);

flow_dict = get_flowdict(0, planar_layers, 0, 0);
flow_ids = flow_dict['flow_ids'];
flow_repeats = flow_dict['flow_repeats'];
print_flowdict(flow_dict);

cost_type = 'KL';
M_eta = 100;
lr_order = -3;
random_seed = 0;
max_iters = 20000;
check_rate = 100;

if (D==2):
	mu = np.array([[1, 5]]);
	Sigma = 5*np.expand_dims(np.array([[1, 0.9], [0.9, 1]]), 0);
	np.random.seed(0);
	N = 100;
	z = np.array([5,1]);
	x = drawPoissonCounts(z, N);
	xs = np.expand_dims(x, 0);
	zs = np.array([z]);
	Ns = np.array([N]);
	xlim = 10;
	ylim = 10;
	"""
	plotContourTruncatedNormal(mu[0], Sigma[0], xlim, ylim, 100, scatter=False);
	plt.figure();
	print(x);
	H, xedges, yedges = np.histogram2d(x[0], x[1], bins=8, range=[[0,xlim],[0,ylim]], normed=True)
	plt.imshow(H.T, origin='lower', interpolation='none');
	plt.show();
	"""
else:
	raise NotImplementedError;


params = {'mus':mu, 'Sigmas':Sigma, 'xs':xs, 'zs':zs, 'Ns':Ns, 'D':D};

log_P, X, R2s, KLs, it = train_nf(exp_fam, params, flow_dict, cost_type, \
		                          M_eta, lr_order, random_seed, max_iters, check_rate);
