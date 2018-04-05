from train_network import train_network
from matplotlib import pyplot as plt

plot_dists = True;
numdists = 125;
flow_id = 'linear1';
n_k = 300;
lr_order = -3;
random_seed = 0;
param_network = False;

for i in range(100,numdists):
	constraint_id = 'normal_%d' % (i+1);
	X, R2s = train_network(constraint_id, flow_id, n_k, lr_order, random_seed, param_network)
	if (plot_dists):
		plt.figure();
		plt.scatter(X[:,0,0], X[:,1,0]);
		plt.title('R2 = %.4f' % R2s[0]);
		plt.show();