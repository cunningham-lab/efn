# Copyright 2018 Sean Bittner, Columbia University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
from efn.train_nf import train_nf
import numpy as np
from scipy.stats import multivariate_normal
import os, sys
from tf_util.families import family_from_str
from efn.util.efn_util import model_opt_hps

os.chdir('../');

exp_fam = str(sys.argv[1]);
D = int(sys.argv[2]);
give_inverse_hint = int(sys.argv[3]) == 1;
dist_seed = int(sys.argv[4]);
random_seed = int(sys.argv[5]);
dir_str = str(sys.argv[6]);

flow_type, nlayers, scale_layer, lr_order = model_opt_hps(exp_fam, D);


fam_class = family_from_str(exp_fam);
family = fam_class(D);

arch_dict = {'D':family.D_Z,
             'K':1,
			 'post_affine':False, \
             'flow_type':flow_type, \
             'repeats':nlayers};

M_eta = 1000;
min_iters = 100000;
max_iters = 1000000;
check_rate = 100;

param_net_input_type = 'eta';  # I should generalize this draw_etas function to accept a None
np.random.seed(dist_seed);
eta, param_net_input, Tx_input, params = family.draw_etas(1, param_net_input_type, give_inverse_hint);
params = params[0];
params.update({'dist_seed':dist_seed});

log_p_zs, X, train_R2s, train_KLs, it = train_nf(family, params, arch_dict, M_eta, lr_order, \
	                         					 random_seed, min_iters, max_iters, check_rate, dir_str);
