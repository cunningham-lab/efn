from train_efn import train_efn
import numpy as np
from scipy.stats import multivariate_normal
import os, sys
from tf_util.families import family_from_str
from efn_util import model_opt_hps

os.chdir("../")

exp_fam = str(sys.argv[1])
D = int(sys.argv[2])
param_net_input_type = str(sys.argv[3])
give_hint = int(sys.argv[4]) == 1
dist_seed = int(sys.argv[5])
random_seed = int(sys.argv[6])
dir_str = str(sys.argv[7])

TIF_flow_type, nlayers, scale_layer, lr_order = model_opt_hps(exp_fam, D)
fam_class = family_from_str(exp_fam)

family = fam_class(D)
flow_dict = {
    "latent_dynamics": None,
    "TIF_flow_type": TIF_flow_type,
    "repeats": nlayers,
    "scale_layer": scale_layer,
}
K = 1
M = 1000
stochastic_eta = False
min_iters = 100000
max_iters = 1000000
check_rate = 100

X, train_KLs, it = train_efn(
    family,
    flow_dict,
    param_net_input_type,
    K,
    M,
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
