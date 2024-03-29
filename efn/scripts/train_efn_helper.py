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
from efn.train_efn import train_efn
import numpy as np
from scipy.stats import multivariate_normal
from tf_util.families import family_from_str
from efn.util.efn_util import model_opt_hps
import os, sys

os.chdir("../")

exp_fam = str(sys.argv[1])
D = int(sys.argv[2])
give_hint = int(sys.argv[3]) == 1
random_seed = int(sys.argv[4])
dir_str = str(sys.argv[5])

flow_type, nlayers, post_affine, lr_order = model_opt_hps(exp_fam, D)
fam_class = family_from_str(exp_fam)

family = fam_class(D)

arch_dict = {
    "D":family.D_Z,
    "K":1,
    "post_affine":post_affine,
    "flow_type": flow_type,
    "repeats": nlayers,
}

param_net_input_type = "eta"
K_eta = 100
M_eta = 1000
stochastic_eta = True
dist_seed = 0
min_iters = 100000
max_iters = 1000000
check_rate = 100

train_efn(
    family,
    arch_dict,
    param_net_input_type,
    K_eta,
    M_eta,
    stochastic_eta,
    give_hint=give_hint,
    lr_order=lr_order,
    dist_seed=dist_seed,
    random_seed=random_seed,
    min_iters=min_iters,
    max_iters=max_iters,
    check_rate=check_rate,
    dir_str=dir_str,
)
