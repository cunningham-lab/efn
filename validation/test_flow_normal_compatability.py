from train_mefn import train_mefn
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import os
import sys
from efn_util import drawEtas
from plot_utils import plotMefnTraining

os.chdir('../');

exp_fam = 'normal';
D = int(sys.argv[1]);
nonlin_spinner_layers = 4;
flow_ids = [];
flow_repeats = [];
for i in range(nonlin_spinner_layers):
	flow_ids.append('StructuredSpinnerLayer');
	flow_repeats.append(1);
	if not (i==(nonlin_spinner_layers-1)):
		flow_ids.append('TanhLayer');
		flow_repeats.append(1);
flow_dict = {'flow_ids':flow_ids, 'flow_repeats':flow_repeats};

def print_flowdict(flow_dict):
	flow_ids = flow_dict['flow_ids'];
	flow_repeats = flow_dict['flow_repeats'];
	nlayers = len(flow_ids);
	for i in range(nlayers):
		print('%d %ss' % (flow_repeats[i], flow_ids[i]));
	return None;

print_flowdict(flow_dict);


cost_type = 'KL';
M_eta = 1000;
lr_order = -2;
random_seed = 0;
check_rate = 100;
max_iters = 20000;
ndraws = 1;

np.random.seed(0);
for i in range(ndraws):
	etas, params = drawEtas(exp_fam, D, 1);
	log_P, X, R2s, KLs, it = train_mefn(exp_fam, params, flow_dict, cost_type, M_eta, lr_order, random_seed, max_iters, check_rate);
	print('xshape', X.shape);
	print('learned in %d iterations' % it)
	#plotMefnTraining(exp_fam, R2s, KLs, X, log_P, params, check_rate, it, 'normal %d' % (i+1));
plt.show();


