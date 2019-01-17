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
import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_kernels
import statsmodels.sandbox.distributions.mv_normal as mvd
import scipy.io as soio
import os
import re
from tf_util.stat_util import get_dist_str
from tf_util.tf_util import get_flowstring
from scipy.stats import (
    ttest_1samp,
    multivariate_normal,
    dirichlet,
    invwishart,
    truncnorm,
)
from tf_util.flows import (
    AffineFlowLayer,
    PlanarFlowLayer,
    RadialFlowLayer,
    SimplexBijectionLayer,
    CholProdLayer,
    StructuredSpinnerLayer,
    StructuredSpinnerTanhLayer,
    TanhLayer,
    ExpLayer,
    SoftPlusLayer,
    GP_EP_CondRegLayer,
    GP_Layer,
    AR_Layer,
    VAR_Layer,
    FullyConnectedFlowLayer,
    ElemMultLayer,
)

p_eps = 10e-6


def get_savedir(
    family,
    model_type_str,
    dir_str,
    param_net_input_type,
    K,
    M,
    flow_dict,
    param_net_hps,
    give_hint,
    random_seed,
    dist_info={},
):
    """Formats the name of the model save directory.

        Args:
            family (obj): Instance of tf_util.families.Family.
            model_type_str (str): Specifies model type (like NF or EFN).
            dir_str (str): Specifiy where to save off off '/efn/models/' filepath.
            param_net_input_type (str): Specifies input to param network.
                'eta':        Give full eta to parameter network.
                'prior':      Part of eta that is prior-dependent.
                'likelihood': Part of eta that is likelihood-dependent.
                'data':       The data itself.
            K (int): Number of distributions per gradient descent batch during training.
            M (int): Number of samples per distribution per batch during training.
            flow_dict (dict): Specifies structure of approximating density network.
            param_net_hps (dict): Parameter network hyperparameters.
            give_hint (bool): Provide hint to parameter network.
            random_seed (int): Tensorflow random seed for initialization.
            dist_info (dict): Information regarding selection of eta sample/prior.

        Returns:
            savedir (str): Path to model save directory.

        """
    resdir = "models/%s/" % dir_str
    eta_str = get_dist_str(family.eta_dist)
    give_hint_str = "giveHint_" if give_hint else ""
    flowstring = get_flowstring(flow_dict)

    if param_net_input_type in ["eta", ""]:
        substr = ""
    elif param_net_input_type == "prior":
        substr = "a"
    elif param_net_input_type == "likelihood":
        substr = "b"
    elif param_net_input_type == "data":
        substr = "c"
    else:
        print(param_net_input_type)
        raise NotImplementedError()

    if model_type_str == "EFN":
        savedir = resdir + "EFN%s_%s%s_%sD=%d_K=%d_M=%d_flow=%s_L=%d_rs=%d/" % (
            substr,
            family.name,
            eta_str,
            give_hint_str,
            family.D,
            K,
            M,
            flowstring,
            param_net_hps["L"],
            random_seed,
        )
    elif model_type_str == "EFN1":
        dist_seed = dist_info["dist_seed"]
        savedir = resdir + "%s%s_%s%s_D=%d_flow=%s_L=%d_ds=%d_rs=%d/" % (
            model_type_str,
            substr,
            family.name,
            eta_str,
            family.D,
            flowstring,
            param_net_hps["L"],
            dist_seed,
            random_seed,
        )
    elif model_type_str == "NF1":
        dist_seed = dist_info["dist_seed"]
        savedir = resdir + "%s%s_%s%s_D=%d_flow=%s_ds=%d_rs=%d/" % (
            model_type_str,
            substr,
            family.name,
            eta_str,
            family.D,
            flowstring,
            dist_seed,
            random_seed,
        )
    return savedir


def model_opt_hps(exp_fam, D):
    """Determine optimization hyperparamters for the choice of family and dimensionality.

        Args:
            exp_fam (str): Name of exponential family model.
            D (int): Dimensionality of exponential family model.

        Returns:
            TIF_flow_type (str): Specification of time-invariant flow network class.
            nlayers (int): Number of layers in the time-invariant flow network.
            scale_layer (bool): An initial element-wise scaling layer is used if True.
            lr_order (float): Order of magnitude of the learning rate for Adam optimizer.

        """
    scale_layer = False

    if exp_fam == "normal":
        TIF_flow_type = "AffineFlowLayer"
        nlayers = 1
        lr_order = -3
    else:
        # flow type
        TIF_flow_type = "PlanarFlowLayer"

        # number of layers
        if exp_fam == "inv_wishart":
            sqrtD = int(np.sqrt(D))
            nlayers = int(sqrtD * (sqrtD + 1) / 2)
        else:
            nlayers = D
        if nlayers < 20:
            nlayers = 20

        # learning rate
        lr_order = -3
        if exp_fam == "dirichlet":
            if D >= 15:
                lr_order = -4
            if D >= 50:
                lr_order = -5

        elif exp_fam == "lgc":
            lr_order = -4
            if D >= 10:
                lr_order = -5

        elif exp_fam == "dir_dir":
            if D >= 10:
                lr_order = -4
            elif D >= 25:
                lr_order = -5

        elif exp_fam == "dir_mult":
            if D >= 10:
                lr_order = -4

        elif exp_fam == "lgc":
            if D >= 15:
                lr_order = -4
            else:
                lr_order = -3

    return TIF_flow_type, nlayers, scale_layer, lr_order


def get_param_network_upl(
    L, num_param_net_inputs, num_theta_params, upl_tau=None, shape="linear"
):
    """Determine units per layer in the parameter network.

        Args:
            L (int): Number of layers in parameter network.
            num_param_net_inputs (int): Dimensionality of parameter network input.
            num_theta_params (int): Number of parameters in the density network.
            upl_tau (float): time-scale of upl trend.
            shape (str): Specification of upl trend class.

        Returns:
            upl_param_net (list): List of units per layer of the parameter network.

        """
    if shape == "linear":
        upl_inc = int(np.floor(abs(num_theta_params - num_param_net_inputs) / (L + 1)))
        upl_param_net = []
        upl_i = min(num_theta_params, num_param_net_inputs)
        for i in range(L):
            upl_i += upl_inc
            upl_param_net.append(upl_i)
    elif shape == "overparam":
        print("overparameterizing the theta network")
        upl_param_net = []
        upl_i = 20
        for i in range(L):
            upl_param_net.append(upl_i)
    elif shape == "exp":
        A = abs(num_theta_params - num_param_net_inputs)
        l = np.arange(L)
        upl = np.exp(l / upl_tau)
        upl = upl - upl[0]
        upl_param_net = np.int32(
            np.round(
                A * ((upl) / upl[-1]) + min(num_theta_params, num_param_net_inputs)
            )
        )

    if num_param_net_inputs > num_theta_params:
        upl_param_net = np.flip(upl_param_net, axis=0)

    return upl_param_net


def construct_param_network(param_net_input, K, flow_layers, param_net_hps):
    """Instantiate and connect layers of the parameter network.

        The parameter network (parameterized by phi in EFN paper) maps natural parameters
        eta to the parameters of the the density network theta.

        Args:
            param_net_input (tf.placeholder): Usually this is a (K X|eta|) tensor holding
                                              eta for each of the K distributions.
                                              Sometimes we provide hints for the parameter
                                              network in addition to eta, which are
                                              concatenated onto the end.
            K (int): Number of distributions.
            flow_layers (list): List of layers of the density network.
            param_net_hps (dict): Parameter network hyperparameters.

        Returns:
            theta (tf.Tensor): Output of the parameter network.

    """

    L_theta = param_net_hps["L"]
    upl_theta = param_net_hps["upl"]
    L_flow = len(flow_layers)
    h = param_net_input
    for i in range(L_theta):
        with tf.variable_scope("ParamNetLayer%d" % (i + 1)):
            h = tf.layers.dense(h, upl_theta[i], activation=tf.nn.tanh)

    out_dim = h.shape[1]
    print(out_dim)

    theta = []
    for i in range(L_flow):
        layer = flow_layers[i]
        layer_name, param_names, param_dims, _, _ = layer.get_layer_info()
        nparams = len(param_names)
        layer_i_params = []
        # read each parameter out of the last layer.
        for j in range(nparams):
            num_elems = np.prod(param_dims[j])
            A_shape = (out_dim, num_elems)
            b_shape = (1, num_elems)
            A_ij = tf.get_variable(
                layer_name + "_" + param_names[j] + "_A",
                shape=A_shape,
                dtype=tf.float64,
                initializer=tf.glorot_uniform_initializer(),
            )
            b_ij = tf.get_variable(
                layer_name + "_" + param_names[j] + "_b",
                shape=b_shape,
                dtype=tf.float64,
                initializer=tf.glorot_uniform_initializer(),
            )
            param_ij = tf.matmul(h, A_ij) + b_ij
            param_ij = tf.reshape(param_ij, (K,) + param_dims[j])
            layer_i_params.append(param_ij)
        theta.append(layer_i_params)
    return theta


def cost_fn(eta, log_p_zs, T_z, log_h_z, K):
    """Compute total cost and ELBOS and r^2 for each distribution.

        Cost is KL_p(eta)(q(z; eta) || p[(z; eta)).

        Args:
            eta (tf.placehoder): (K x |eta|) tensor of eta for each distribution.
            log_p_zs (tf.Tensor): (K x M) tensor of log probability of each sample.
            T_z (tf.Tensor): (K x M x |T(z)|) tensor of suff statistics of each sample.
            log_h_z (tf.Tensor): (K x M x 1) tensor of log base measure of each sample.
            K (int): Number of distributions.

        Returns:
            cost (tf.Tensor): Scalar model approximation loss.
            elbos (list): List of tf.Tensor ELBOs of each distribution.
            R2s (list): List of tf.Tensor r^2s of each distribution.

        """

    y = log_p_zs
    R2s = []
    elbos = []
    for k in range(K):
        # get eta-specific log-probs and T(x)'s
        y_k = tf.expand_dims(y[k, :], 1)
        T_z_k = T_z[k, :, :]
        log_h_z_k = tf.expand_dims(log_h_z[k, :], 1)
        eta_k = tf.expand_dims(eta[k, :], 1)
        # compute optimial linear regression offset term for eta
        alpha_k = tf.reduce_mean(y_k - (tf.matmul(T_z_k, eta_k) + log_h_z_k))
        residuals = y_k - (tf.matmul(T_z_k, eta_k) + log_h_z_k) - alpha_k
        RSS_k = tf.matmul(tf.transpose(residuals), residuals)
        y_k_mc = y_k - tf.reduce_mean(y_k)
        TSS_k = tf.reduce_sum(tf.square(y_k_mc))
        # compute the R^2 of the exponential family fit
        R2s.append(1.0 - (RSS_k[0, 0] / TSS_k))
        elbos.append(-tf.reduce_mean(y_k - (tf.matmul(T_z_k, eta_k) + T_z_k)))

    y = tf.expand_dims(log_p_zs, 2)
    log_h_z = tf.expand_dims(log_h_z, 2)
    eta = tf.expand_dims(eta, 2)
    # we want to minimize the mean negative elbo
    cost = tf.reduce_sum(tf.reduce_mean(y - (tf.matmul(T_z, eta) + log_h_z), [1, 2]))
    return cost, elbos, R2s


def setup_param_logging(all_params):
    """Setup tensorboard parameter logging

        Args:
            all_params (list): List of all tf.Variable parameters of model.

        """
    nparams = len(all_params)
    for i in range(nparams):
        param = all_params[i]
        param_shape = tuple(param.get_shape().as_list())
        for ii in range(param_shape[0]):
            if len(param_shape) == 1 or (len(param_shape) < 2 and param_shape[1] == 1):
                tf.summary.scalar("%s_%d" % (param.name[:-2], ii + 1), param[ii])
            else:
                for jj in range(param_shape[1]):
                    tf.summary.scalar(
                        "%s_%d%d" % (param.name[:-2], ii + 1, jj + 1), param[ii, jj]
                    )
    return None
    

def autocovariance(X, tau_max, T, batch_size):
    # need to finish this
    X_toep = []
    X_toep1 = []
    X_toep2 = []
    for i in range(tau_max + 1):
        X_toep.append(X[:, :, i : ((T - tau_max) + i)])
        # This will be (n x D x tau_max x (T- tau_max))
        X_toep1.append(X[: (batch_size // 2), :, i : ((T - tau_max) + i)])
        X_toep2.append(X[(batch_size // 2) :, :, i : ((T - tau_max) + i)])

    X_toep = tf.reshape(
        tf.transpose(tf.convert_to_tensor(X_toep), [2, 0, 3, 1]),
        [D, tau_max + 1, batch_size * (T - tau_max)],
    )
    # D x tau_max x (T-tau_max)*n
    X_toep1 = tf.reshape(
        tf.transpose(tf.convert_to_tensor(X_toep1), [2, 0, 3, 1]),
        [D, tau_max + 1, (batch_size // 2) * (T - tau_max)],
    )
    # D x tau_max x (T-tau_max)*n
    X_toep2 = tf.reshape(
        tf.transpose(tf.convert_to_tensor(X_toep2), [2, 0, 3, 1]),
        [D, tau_max + 1, (batch_size // 2) * (T - tau_max)],
    )
    # D x tau_max x (T-tau_max)*n

    X_toep_mc = X_toep - tf.expand_dims(tf.reduce_mean(X_toep, 2), 2)
    X_toep_mc1 = X_toep1 - tf.expand_dims(tf.reduce_mean(X_toep1, 2), 2)
    X_toep_mc2 = X_toep2 - tf.expand_dims(tf.reduce_mean(X_toep2, 2), 2)

    X_tau = (
        tf.cast((1 / (batch_size * (T - tau_max))), tf.float64)
        * tf.matmul(X_toep_mc[:, :, :], tf.transpose(X_toep_mc, [0, 2, 1]))[:, :, 0]
    )
    X_tau1 = (
        tf.cast((1 / ((batch_size // 2) * (T - tau_max))), tf.float64)
        * tf.matmul(X_toep_mc1[:, :, :], tf.transpose(X_toep_mc1, [0, 2, 1]))[:, :, 0]
    )
    X_tau2 = (
        tf.cast((1 / ((batch_size // 2) * (T - tau_max))), tf.float64)
        * tf.matmul(X_toep_mc2[:, :, :], tf.transpose(X_toep_mc2, [0, 2, 1]))[:, :, 0]
    )

    X_tau_err = tf.reshape(
        X_tau - autocov_targ[:, : (tau_max + 1)], (D * (tau_max + 1),)
    )
    X_tau_err1 = tf.reshape(
        X_tau1 - autocov_targ[:, : (tau_max + 1)], (D * (tau_max + 1),)
    )
    X_tau_err2 = tf.reshape(
        X_tau2 - autocov_targ[:, : (tau_max + 1)], (D * (tau_max + 1),)
    )
    tau_MSE = tf.reduce_sum(tf.square(X_tau_err))
    Tx_autocov = 0
    Rx_autocov = 0
    return Tx_autocov, Rx_autocov


def adam_updates(params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999):
    """ Adam optimizer """
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads
    t = tf.Variable(1.0, "adam_t")
    for p, g in zip(params, grads):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + "_adam_mg", dtype=tf.float64)
        if mom1 > 0:
            v = tf.Variable(
                tf.zeros(p.get_shape()), p.name + "_adam_v", dtype=tf.float64
            )
            v_t = mom1 * v + (1.0 - mom1) * g
            v_hat = v_t / (1.0 - tf.pow(mom1, t))
            updates.append(v.assign(v_t))
        else:
            v_hat = g
        mg_t = mom2 * mg + (1.0 - mom2) * tf.square(g)
        mg_hat = mg_t / (1.0 - tf.pow(mom2, t))
        g_t = v_hat / tf.sqrt(mg_hat + 1e-8)
        p_t = p - lr * g_t
        updates.append(mg.assign(mg_t))
        updates.append(p.assign(p_t))
    updates.append(t.assign_add(1))
    return tf.group(*updates)


def compute_VAR_cov_tf(As, Sigma_eps, D, K, T):
    # initialize the covariance matrix
    zcov = [[tf.eye(D)]]

    # compute the lagged noise-state covariance
    gamma = [Sigma_eps]
    for t in range(1, T):
        gamma_t = tf.zeros((D, D), dtype=tf.float64)
        for k in range(1, min(t, K) + 1):
            gamma_t += tf.matmul(As[k - 1], gamma[t - k])
        gamma.append(gamma_t)

    # compute the off-block-diagonal covariance blocks
    # first row
    for s in range(1, T):
        zcov_0s = tf.zeros((D, D), dtype=tf.float64)
        for k in range(1, min(s, K) + 1):
            tau = s - k
            zcov_0tau = zcov[0][tau]
            zcov_0s += tf.matmul(zcov_0tau, tf.transpose(As[k - 1]))
        zcov[0].append(zcov_0s)
        zcov.append([tf.transpose(zcov_0s)])

    # remaining rows
    for t in range(1, T):
        for s in range(t, T):
            zcov_ts = tf.zeros((D, D), dtype=tf.float64)
            # compute the contribution of lagged state-state covariances
            for k_t in range(1, min(t, K) + 1):
                tau_t = t - k_t
                for k_s in range(1, min(s, K) + 1):
                    tau_s = s - k_s
                    zcov_tauttaus = zcov[tau_t][tau_s]
                    zcov_ts += tf.matmul(
                        As[k_t - 1], tf.matmul(zcov_tauttaus, tf.transpose(As[k_s - 1]))
                    )
            # compute the contribution of lagged noise-state covariances
            if t == s:
                zcov_ts += Sigma_eps
            for k in range(1, min(s, K) + 1):
                tau_s = s - k
                if tau_s >= t:
                    zcov_ts += tf.matmul(
                        tf.transpose(gamma[tau_s - t]), tf.transpose(As[k - 1])
                    )

            zcov[t].append(zcov_ts)
            if t != s:
                zcov[s].append(tf.transpose(zcov_ts))

    zcov = tf.convert_to_tensor(zcov)
    Zcov = tf.reshape(tf.transpose(zcov, [0, 2, 1, 3]), (D * T, D * T))
    return Zcov


def compute_VAR_cov_np(As, Sigma_eps, D, K, T):
    # Compute the analytic covariance of the VAR model

    # initialize the covariance matrix
    zcov = np.zeros((D * T, D * T))

    # compute the block-diagonal covariance
    zcov[:D, :D] = np.eye(D)

    # compute the lagged noise-state covariance
    gamma = [Sigma_eps]
    for t in range(1, T):
        gamma_t = np.zeros((D, D))
        for k in range(1, min(t, K) + 1):
            gamma_t += np.dot(As[k - 1], gamma[t - k])
        gamma.append(gamma_t)

    # compute the off-block-diagonal covariance blocks
    # first row
    for s in range(1, T):
        zcov_0s = np.zeros((D, D))
        for k in range(1, min(s, K) + 1):
            tau = s - k
            zcov_0tau = zcov[:D, (D * tau) : (D * (tau + 1))]
            zcov_0s += np.dot(zcov_0tau, As[k - 1].T)
        zcov[:D, (D * s) : (D * (s + 1))] = zcov_0s
        zcov[(D * s) : (D * (s + 1)), :D] = zcov_0s.T

    # remaining rows
    for t in range(1, T):
        for s in range(t, T):
            zcov_ts = np.zeros((D, D))
            # compute the contribution of lagged state-state covariances
            for k_t in range(1, min(t, K) + 1):
                tau_t = t - k_t
                for k_s in range(1, min(s, K) + 1):
                    tau_s = s - k_s
                    zcov_tauttaus = zcov[
                        (D * tau_t) : (D * (tau_t + 1)), (D * tau_s) : (D * (tau_s + 1))
                    ]
                    zcov_ts += np.dot(As[k_t - 1], np.dot(zcov_tauttaus, As[k_s - 1].T))

            # compute the contribution of lagged noise-state covariances
            if s == t:
                zcov_ts += Sigma_eps
            for k in range(1, min(s, K) + 1):
                tau_s = s - k
                if tau_s >= t:
                    zcov_ts += np.dot(gamma[tau_s - t].T, As[k - 1].T)

            zcov[(D * t) : (D * (t + 1)), (D * s) : (D * (s + 1))] = zcov_ts
            zcov[(D * s) : (D * (s + 1)), (D * t) : (D * (t + 1))] = zcov_ts.T
    return zcov


def simulate_VAR(As, Sigma_eps, T):
    K = As.shape[0]
    D = As.shape[1]
    mu = np.zeros((D,))
    z = np.zeros((D, T))
    z[:, 0] = np.random.multivariate_normal(mu, np.eye(D))
    for t in range(1, T):
        Z_VAR_pred_t = np.zeros((D,))
        for k in range(K):
            if t - (k + 1) >= 0:
                Z_VAR_pred_t += np.dot(As[k], z[:, t - (k + 1)])
        eps_t = np.random.multivariate_normal(mu, Sigma_eps)
        z[:, t] = Z_VAR_pred_t + eps_t
    return z


def computeGPcov(l, T, eps):
    D = l.shape[0]
    for d in range(D):
        diffmat = np.zeros((T, T), dtype=np.float64)
        for i in range(T):
            for j in range(T):
                diffmat_ij = np.float64(i - j)
                diffmat[i, j] = diffmat_ij
                if i is not j:
                    diffmat[j, i] = diffmat_ij
        GPcovd = np.exp(-np.square(diffmat) / (2.0 * np.square(l[0]))) + eps * np.eye(
            T, dtype=np.float64
        )
        L = np.linalg.cholesky(GPcovd)
        if d == 0:
            GPcov = np.expand_dims(GPcovd, 0)
        else:
            GPcov = np.concatenate((GPcov, np.expand_dims(GPcovd, 0)), axis=0)
    return GPcov


def sampleGP(GPcov, n):
    D = GPcov.shape[0]
    T = GPcov.shape[1]
    for d in range(D):
        GPcovd = GPcov[d, :, :]
        L = np.linalg.cholesky(GPcovd)
        Z0d = np.dot(L, np.random.normal(np.zeros((T, n)), 1.0))
        if d == 0:
            Z_GP = np.expand_dims(Z0d, 0)
        else:
            Z_GP = np.concatenate((Z_GP, np.expand_dims(Z0d, 0)), axis=0)
    return Z_GP


# Gabriel's MMD stuff
def MMD2u(K, m, n):
    """The MMD^2_u unbiased statistic.
    """
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    return (
        1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum())
        + 1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum())
        - 2.0 / (m * n) * Kxy.sum()
    )


def compute_null_distribution(
    K, m, n, iterations=10000, verbose=False, random_state=None, marker_interval=1000
):
    """Compute the bootstrap null-distribution of MMD2u.
    """
    if type(random_state) == type(np.random.RandomState()):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        idx = rng.permutation(m + n)
        K_i = K[idx, idx[:, None]]
        mmd2u_null[i] = MMD2u(K_i, m, n)

    if verbose:
        print("")

    return mmd2u_null


def kernel_two_sample_test(
    X,
    Y,
    kernel_function="rbf",
    iterations=10000,
    verbose=False,
    random_state=None,
    **kwargs
):
    """Compute MMD^2_u, its null distribution and the p-value of the
    kernel two-sample test.
    Note that extra parameters captured by **kwargs will be passed to
    pairwise_kernels() as kernel parameters. E.g. if
    kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1),
    then this will result in getting the kernel through
    kernel_function(metric='rbf', gamma=0.1).
    """
    m = len(X)
    n = len(Y)
    XY = np.vstack([X, Y])
    K = pairwise_kernels(XY, metric=kernel_function, **kwargs)
    mmd2u = MMD2u(K, m, n)
    if verbose:
        print("MMD^2_u = %s" % mmd2u)
        print("Computing the null distribution.")

    mmd2u_null = compute_null_distribution(
        K, m, n, iterations, verbose=verbose, random_state=random_state
    )
    p_value = max(1.0 / iterations, (mmd2u_null > mmd2u).sum() / float(iterations))
    if verbose:
        print("p-value ~= %s \t (resolution : %s)" % (p_value, 1.0 / iterations))

    return mmd2u, mmd2u_null, p_value


def check_convergence(to_check, cur_ind, lag, thresh, criteria="lag_diff", wsize=500):
    len_to_check = len(to_check)
    vals = to_check[0]
    for i in range(1, len_to_check):
        vals = np.concatenate((vals, to_check[1]), axis=1)

    if criteria == "lag_diff":
        lag_mean = np.mean(vals[(cur_ind - (lag + wsize)) : (cur_ind - lag), :], axis=0)
        cur_mean = np.mean(vals[(cur_ind - wsize) : cur_ind, :], axis=0)
        log_param_diff = np.log(np.linalg.norm(lag_mean - cur_mean))
        has_converged = log_param_diff < thresh
    elif criteria == "grad_mean_ttest":
        last_grads = vals[(cur_ind - lag) : cur_ind, :]
        # Sigma_grads = np.dot(last_grads.T, last_grads) / (lag); # zero-mean covariance
        nvars = last_grads.shape[1]
        # mvt = mvd.MVT(np.zeros((nvars,)), Sigma_grads, lag);
        # grad_mean = np.mean(last_grads, 0);
        # t_cdf = mvt.cdf(grad_mean);
        # has_converged = (t_cdf > (thresh/2) and t_cdf < (1-(thresh/2)));
        # print('cdf val', t_cdf, 'convergence', has_converged);
        has_converged = True
        for i in range(nvars):
            t, p = ttest_1samp(last_grads[:, i], 0)
            # if any grad mean is not zero, reject
            if p < thresh:
                has_converged = False
                break
    return has_converged


def test_convergence(mean_test_elbos, ind, wsize, delta_thresh):
    cur_mean_test_elbo = np.mean(mean_test_elbos[(ind - wsize + 1) : (ind + 1)])
    prev_mean_test_elbo = np.mean(
        mean_test_elbos[(ind - 2 * wsize + 1) : (ind - wsize + 1)]
    )
    delta_elbo = (prev_mean_test_elbo - cur_mean_test_elbo) / prev_mean_test_elbo
    return delta_elbo < delta_thresh


def find_convergence(mean_test_elbos, last_ind, wsize, delta_thresh):
    for ind in range(wsize, last_ind + 1):
        if test_convergence(mean_test_elbos, ind, wsize, delta_thresh):
            return ind
    return None


def factors(n):
    return [f for f in range(1, n + 1) if n % f == 0]


def easy_inds():
    num_monkeys = 3
    num_neurons = [83, 59, 105]
    num_oris = 12
    N = int(sum(num_neurons * num_oris))
    monkeys = np.zeros((N,))
    neurons = np.zeros((N,))
    oris = np.zeros((N,))

    ind = 0
    for i in range(num_monkeys):
        monkey = i + 1
        nneurons = num_neurons[i]
        for j in range(nneurons):
            neuron = j + 1
            for k in range(num_oris):
                ori = k + 1
                monkeys[ind] = monkey
                neurons[ind] = neuron
                oris[ind] = ori
                ind += 1
    return monkeys, neurons, oris


def log_grads(cost_grads, cost_grad_vals, ind):
    cgv_ind = 0
    nparams = len(cost_grads)
    for i in range(nparams):
        grad = cost_grads[i]
        grad_shape = grad.shape
        ngrad_vals = np.prod(grad_shape)
        grad_reshape = np.reshape(grad, (ngrad_vals,))
        for ii in range(ngrad_vals):
            cost_grad_vals[ind, cgv_ind] = grad_reshape[ii]
            cgv_ind += 1
    return None
