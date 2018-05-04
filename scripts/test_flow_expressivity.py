from train_mefn import train_mefn
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import os
import sys
from efn_util import drawEtas

os.chdir('../');

exp_fam = str(sys.argv[1]);
fully_connected_layers = int(sys.argv[2]);
planar_layers = int(sys.argv[3]);
spinner_layers = int(sys.argv[4]);
nonlin_spinner_layers = int(sys.argv[5]);
flow_ids = [];
flow_repeats = [];

if (fully_connected_layers):
	flow_ids.append('LinearFlowLayer');
	flow_repeats.append(1); # no reason to have more than one here

if (planar_layers > 0):
	flow_ids.append('PlanarFlowLayer');
	flow_repeats.append(planar_layers);

if (spinner_layers > 0):
	flow_ids.append('StructuredSpinnerLayer');
	flow_repeats.append(spinner_layers);

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

def get_flowstring(flow_dict):
	flow_ids = flow_dict['flow_ids'];
	flow_repeats = flow_dict['flow_repeats'];
	nlayers = len(flow_ids);
	flowidstring = '';
	for i in range(nlayers):
		flowidstring += '%d%s' % (flow_repeats[i], flow_ids[i][0]);
		if (i < (nlayers-1)):
			flowidstring += '_';
	return flowidstring;

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
		etas, params = drawEtas(exp_fam, D, 1);
		log_P, X, R2s_ij, KLs_ij, it = train_mefn(exp_fam, params, flow_dict, cost_type, M_eta, lr_order, random_seed, max_iters, check_rate);
		R2s[i,j] = R2s_ij[-1,0];
		KLs[i,j] = KLs_ij[-1,0];
	print('saving', fname);
	np.savez(fname, R2s=R2s, KLs=KLs, Ds=Ds);
print('done');


