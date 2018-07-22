import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_kernels
from scipy.stats import ttest_1samp, multivariate_normal, dirichlet, invwishart, truncnorm
import statsmodels.sandbox.distributions.mv_normal as mvd
import matplotlib.pyplot as plt
from flows import AffineFlowLayer, PlanarFlowLayer, RadialFlowLayer, SimplexBijectionLayer, \
                  CholProdLayer, StructuredSpinnerLayer, StructuredSpinnerTanhLayer, TanhLayer, ExpLayer, \
                  SoftPlusLayer, GP_EP_CondRegLayer, GP_EP_CondRegFillLayer, GP_Layer, AR_Layer, VAR_Layer, \
                  FullyConnectedFlowLayer, ElemMultLayer
import scipy.io as sio
import os
import re

p_eps = 10e-6;

def setup_IO(family, model_type_str, dir_str, param_net_input_type, K, M, flow_dict, \
             param_net_hps, stochastic_eta, give_hint, random_seed, dist_info={}):
    # set file I/O stuff
    resdir = 'results/%s/' % dir_str;
    eta_str = 'stochasticEta' if stochastic_eta else 'fixedEta';
    give_hint_str = 'giveHint_' if give_hint else '';
    flowstring = get_flowstring(flow_dict);

    if (param_net_input_type == 'eta'):
        substr = '' ;
    elif (param_net_input_type == 'prior'):
        substr = 'a';
    elif (param_net_input_type == 'likelihood'):
        substr = 'b';
    elif (param_net_input_type == 'data'):
        substr = 'c';
    else:
        print(param_net_input_type);
        raise NotImplementedError();

    if (model_type_str == 'EFN'):
        savedir = resdir + 'EFN%s_%s_%s_%sD=%d_K=%d_M=%d_flow=%s_L=%d_rs=%d/' \
                  % (substr, family.name, eta_str, give_hint_str, family.D, K, M, flowstring, param_net_hps['L'], random_seed);
    else:
        dist_seed = dist_info['dist_seed'];
        savedir = resdir + '%s%s_%s_D=%d_flow=%s_ds=%d_rs=%d/' % (model_type_str, substr, family.name, family.D, flowstring, dist_seed, random_seed);
    return savedir

def model_opt_hps(exp_fam, D):
    if (exp_fam == 'normal'):
        TIF_flow_type = 'AffineFlowLayer';
        nlayers = 1;
        lr_order = -3;
    else:
        # flow type
        TIF_flow_type = 'PlanarFlowLayer';

        # number of layers
        if (exp_fam == 'inv_wishart'):  
            sqrtD = int(np.sqrt(D));
            nlayers = int(sqrtD*(sqrtD+1)/2);
        else:
            nlayers = D;
        if (nlayers < 20):
            nlayers = 20;

        # learning rate
        lr_order = -3;
        if (exp_fam == 'dirichlet' or exp_fam == 'dir_dir'):
            if (D >= 15):
                lr_order = -4;
        elif (exp_fam == 'dir_mult'):
            if (D >= 10):
                lr_order = -4;

    return TIF_flow_type, nlayers, lr_order;


def get_param_network_hyperparams(L, num_param_net_inputs, num_theta_params, upl_tau, shape='linear'):
    if (shape=='linear'):
        upl_inc = int(np.floor(abs(num_theta_params - num_param_net_inputs) / (L + 1)));
        upl_param_net = [];
        upl_i = min(num_theta_params, num_param_net_inputs);
        for i in range(L):
            upl_i += upl_inc;
            upl_param_net.append(upl_i);
    elif (shape=='exp'):
        A = abs(num_theta_params-num_param_net_inputs);
        l = np.arange(L);
        upl = np.exp(l/upl_tau);
        upl = upl - upl[0];
        upl_param_net = np.int32(np.round(A*((upl) / upl[-1]) + min(num_theta_params, num_param_net_inputs)));

    if (num_param_net_inputs > num_theta_params):
        upl_param_net = np.flip(upl_param_net, axis=0);

    print(num_param_net_inputs, '->', num_theta_params);
    print(upl_param_net, 'sum', sum(upl_param_net));
    param_net_hps = {'L':L, 'upl':upl_param_net};
    return param_net_hps;

def construct_flow(flow_dict, D_Z, T):
    latent_layers = construct_latent_dynamics(flow_dict, D_Z, T);
    time_invariant_layers = construct_time_invariant_flow(flow_dict, D_Z, T);

    layers = latent_layers + time_invariant_layers;
    nlayers = len(layers);

    num_theta_params = 0;
    for i in range(nlayers):
        layer = layers[i];
        print(i, layer);
        num_theta_params += count_layer_params(layer);

    Z0 = tf.placeholder(tf.float64, shape=(None, None, D_Z, None), name='Z0');
    K = tf.shape(Z0)[0];
    M = tf.shape(Z0)[1];

    p0 = tf.reduce_prod(tf.exp((-tf.square(Z0))/2.0)/np.sqrt(2.0*np.pi), axis=[2,3]); 
    base_log_q_x = tf.log(p0[:,:]);
    Z_AR = Z0;
    return layers, Z0, Z_AR, base_log_q_x, num_theta_params;

def construct_latent_dynamics(flow_dict, D_Z, T):
    latent_dynamics = flow_dict['latent_dynamics'];

    if (latent_dynamics is None):
        return [];

    inits = flow_dict['inits'];
    if ('lock' in flow_dict):
        lock = flow_dict['lock'];
    else:
        lock = False;

    if (latent_dynamics == 'GP'):
        layer = GP_Layer('GP_Layer', dim=D_Z, \
                         inits=inits, lock=lock);

    elif (latent_dynamics == 'AR'):
        param_init = {'alpha_init':inits['alpha_init'], 'sigma_init':inits['sigma_init']};
        layer = AR_Layer('AR_Layer', dim=D_Z, T=T, P=flow_dict['P'], \
                      inits=inits, lock=lock);

    elif (latent_dynamics == 'VAR'):
        param_init = {'A_init':inits['A_init'], 'sigma_init':inits['sigma_init']};
        layer = VAR_Layer('VAR_Layer', dim=D_Z, T=T, P=flow_dict['P'], \
                      inits=inits, lock=lock);

    else:
        raise NotImplementedError();

    return [layer];


def construct_time_invariant_flow(flow_dict, D_Z, T):
    layer_ind = 1;
    layers = [];
    TIF_flow_type = flow_dict['TIF_flow_type'];
    repeats = flow_dict['repeats'];

    if (TIF_flow_type == 'ScalarFlowLayer'):
        flow_class = ElemMultLayer;
        name_prefix = 'ScalarFlow_Layer';

    elif (TIF_flow_type == 'FullyConnectedFlowLayer'):
        flow_class = FullyConnectedFlowLayer;
        name_prefix = FullyConnectedFlow_Layer;

    elif (TIF_flow_type == 'AffineFlowLayer'):
        flow_class = AffineFlowLayer;
        name_prefix = 'AffineFlow_Layer';

    elif (TIF_flow_type == 'StructuredSpinnerLayer'):
        flow_class = StructuredSpinnerLayer
        name_prefix = 'StructuredSpinner_Layer';

    elif (TIF_flow_type == 'StructuredSpinnerTanhLayer'):
        flow_class = StructuredSpinnerTanhLayer
        name_prefix = 'StructuredSpinnerTanh_Layer';

    elif (TIF_flow_type == 'PlanarFlowLayer'):
        flow_class = PlanarFlowLayer
        name_prefix = 'PlanarFlow_Layer';

    elif (TIF_flow_type == 'RadialFlowLayer'):
        flow_class = RadialFlowLayer
        name_prefix = 'RadialFlow_Layer';

    elif (TIF_flow_type == 'TanhLayer'):
        flow_class = TanhLayer;
        name_prefix = 'Tanh_Layer';

    else:
        raise NotImplementedError();

    for i in range(repeats):
        layers.append(flow_class('%s%d' % (name_prefix, layer_ind), D_Z));
        layer_ind += 1;
        
    return layers;

def construct_param_network(param_net_input, K_eta, flow_layers, param_net_hps):
    L_theta = param_net_hps['L']
    upl_theta = param_net_hps['upl'];
    L_flow = len(flow_layers);
    h = param_net_input;
    for i in range(L_theta):
        with tf.variable_scope('ParamNetLayer%d' % (i+1)):
            h = tf.layers.dense(h, upl_theta[i], activation=tf.nn.tanh);
    theta = [];
    for i in range(L_flow):
        layer = flow_layers[i];
        layer_name, param_names, param_dims, _, _ = layer.get_layer_info();
        nparams = len(param_names);
        layer_i_params = [];
        # read each parameter out of the last layer.
        for j in range(nparams):
            num_elems = np.prod(param_dims[j]);
            A_shape = (upl_theta[-1], num_elems);
            b_shape = (1, num_elems);
            A_ij = tf.get_variable(layer_name+'_'+param_names[j]+'_A', shape=A_shape, \
                                       dtype=tf.float64, \
                                       initializer=tf.glorot_uniform_initializer());
            b_ij = tf.get_variable(layer_name+'_'+param_names[j]+'_b', shape=b_shape, \
                                       dtype=tf.float64, \
                                       initializer=tf.glorot_uniform_initializer());
            param_ij = tf.matmul(h, A_ij) + b_ij;
            #param_ij = tf.reshape(param_ij, (K,) + param_dims[j]);
            param_ij = tf.reshape(param_ij, (K_eta,) + param_dims[j]);
            layer_i_params.append(param_ij);
        theta.append(layer_i_params);
    return theta;

def declare_theta(flow_layers):
    L_flow = len(flow_layers);
    theta =[];
    for i in range(L_flow):
        layer = flow_layers[i];
        layer_name, param_names, param_dims, initializers, lock = layer.get_layer_info();
        nparams = len(param_names);
        layer_i_params = [];
        for j in range(nparams):
            if (lock):
                param_ij = initializers[j];
            else:
                if (isinstance(initializers[j], tf.Tensor)):
                    param_ij = tf.get_variable(layer_name+'_'+param_names[j], \
                                               dtype=tf.float64, \
                                               initializer=initializers[j]);
                else:
                    param_ij = tf.get_variable(layer_name+'_'+param_names[j], shape=param_dims[j], \
                                               dtype=tf.float64, \
                                               initializer=initializers[j]);
            layer_i_params.append(param_ij);
        theta.append(layer_i_params);
    return theta;

def count_layer_params(layer):
    num_params = 0;
    name, param_names, dims, _, _ = layer.get_layer_info();
    nparams = len(dims);
    for j in range(nparams):
        num_params += np.prod(dims[j]);
    return num_params;

def latent_dynamics(Z0, A, sigma_eps, T):
    P = A.shape[0];
    D = tf.shape(A)[1];
    K = tf.shape(Z0)[0];
    M = tf.shape(Z0)[1];
    batch_size = tf.multiply(K,M);
    Z_AR = Z0;
    for t in range(1, T):
        Z_AR_pred_t = tf.zeros((K, M, D), dtype=tf.float64);
        for i in range(P):
            if (t-(i+1) >= 0):
                Z_AR_pred_t += tf.transpose(tf.tensordot(A[i,:,:], Z_AR[:,:,:,t-(i+1)], [[1], [2]]), [1,2,0]);
        epsilon = tf.transpose(tf.multiply(sigma_eps, tf.random_normal((D,K,M),0,1)), [1,2,0]);
        Z_AR_t = Z_AR_pred_t + epsilon;

        Z_AR = tf.concat((Z_AR, tf.expand_dims(Z_AR_t, 3)), axis=3);

    Sigma_eps = tf.diag(tf.square(sigma_eps[:,0]));
    Sigma_Z_AR = compute_VAR_cov_tf(A, Sigma_eps, D, P, T);
    Sigma_Z_AR = Sigma_Z_AR;
    
    Z_AR_dist = tf.contrib.distributions.MultivariateNormalFullCovariance(tf.zeros((D*T,), dtype=tf.float64), Sigma_Z_AR);

    Z_AR_dist_shaped = tf.reshape(tf.transpose(Z_AR, [0, 1, 3, 2]), (K,M,D*T));
    log_p_Z_AR = Z_AR_dist.log_prob(Z_AR_dist_shaped);
    return Z_AR, log_p_Z_AR, Sigma_Z_AR;

def connect_flow(Z, layers, theta, ts=None):
    Z_shape = tf.shape(Z);
    K = Z_shape[0];
    M = Z_shape[1];
    D_Z = Z_shape[2];
    T = Z_shape[3];

    sum_log_det_jacobians = tf.zeros((K,M), dtype=tf.float64);
    nlayers = len(layers);
    Z_by_layer = [];
    Z_by_layer.append(Z);
    print('zshapes in');
    print('connect flow');
    for i in range(nlayers):
        print(Z.shape);
        layer = layers[i];
        print(i, layer.name);
        theta_layer = theta[i];
        layer.connect_parameter_network(theta_layer);
        if (isinstance(layer, GP_Layer) or isinstance(layer, GP_EP_CondRegLayer)):
            Z, sum_log_det_jacobians = layer.forward_and_jacobian(Z, sum_log_det_jacobians, ts);
        else:
            Z, sum_log_det_jacobians = layer.forward_and_jacobian(Z, sum_log_det_jacobians);
        Z_by_layer.append(Z);
    print(Z.shape);
    return Z, sum_log_det_jacobians, Z_by_layer;

def cost_fn(eta, log_p_zs, T_x, log_h_x, K_eta, cost_type):
    y = log_p_zs;
    R2s = [];
    elbos = [];
    for k in range(K_eta):
        # get eta-specific log-probs and T(x)'s
        y_k = tf.expand_dims(y[k,:], 1);
        T_x_k = T_x[k,:,:];
        log_h_x_k = tf.expand_dims(log_h_x[k,:], 1);
        eta_k = tf.expand_dims(eta[k,:], 1);
        # compute optimial linear regression offset term for eta
        alpha_k = tf.reduce_mean(y_k - (tf.matmul(T_x_k, eta_k) + log_h_x_k));
        residuals = y_k - (tf.matmul(T_x_k, eta_k)+log_h_x_k) - alpha_k;
        RSS_k = tf.matmul(tf.transpose(residuals), residuals);
        y_k_mc = y_k - tf.reduce_mean(y_k);
        TSS_k = tf.reduce_sum(tf.square(y_k_mc));
        # compute the R^2 of the exponential family fit
        R2s.append(1.0 - (RSS_k[0,0] / TSS_k));
        elbos.append(tf.reduce_mean(y_k - (tf.matmul(T_x_k, eta_k)+log_h_x_k)));

    y = tf.expand_dims(log_p_zs, 2);
    log_h_x = tf.expand_dims(log_h_x, 2);
    eta = tf.expand_dims(eta, 2);
    cost = tf.reduce_sum(tf.reduce_mean(y - (tf.matmul(T_x, eta) + log_h_x), [1,2]));
    #return cost, costs, R2s;
    return cost, elbos, R2s;


def drawPoissonRates(D, ratelim):
    return np.random.uniform(0.1, ratelim, (D,));

def drawPoissonCounts(z, N):
    D = z.shape[0];
    x = np.zeros((D,N));
    for i in range(D):
        x[i,:] = np.random.poisson(z[i], (N,));
    return x;

def truncated_multivariate_normal_rvs(mu, Sigma):
    D = mu.shape[0];
    L = np.linalg.cholesky(Sigma);
    rejected = True;
    count = 1;
    while (rejected):
        z0 = np.random.normal(0,1,(D));
        z = np.dot(L, z0) + mu;
        rejected = 1 - np.prod((np.sign(z)+1)/2);
        count += 1;
    return z;

def get_GP_Sigma(tau, T, Ts):
    K = np.zeros((T, T));
    for i in range(T):
        for j in range(i,T):
            diff = (i-j)*Ts;
            K[i,j] = np.exp(-(np.abs(diff)**2) / (2*(tau**2)));
            if (i != j):
                K[j,i] = K[i,j];
    return K;
    
def get_flowstring(flow_dict):
    latent_dynamics = flow_dict['latent_dynamics'];
    tif_flow_type = flow_dict['TIF_flow_type'];
    repeats = flow_dict['repeats'];
    tif_str = '%d%s' % (repeats, tif_flow_type[:1]);
    if (latent_dynamics is not None):
        return '%s_%s' % (latent_dynamics, tif_str);
    else:
        return tif_str;

def setup_param_logging(all_params):
    nparams = len(all_params);
    for i in range(nparams):
        param = all_params[i];
        param_shape = tuple(param.get_shape().as_list());
        for ii in range(param_shape[0]):
            if (len(param_shape)==1 or (len(param_shape) < 2 and param_shape[1]==1)):
                tf.summary.scalar('%s_%d' % (param.name[:-2], ii+1), param[ii]);
            else:
                for jj in range(param_shape[1]):
                    tf.summary.scalar('%s_%d%d' % (param.name[:-2], ii+1, jj+1), param[ii, jj]);
    return None;

def count_params(all_params):
    nparams = len(all_params);
    nparam_vals = 0;
    for i in range(nparams):
        param = all_params[i];
        param_shape = tuple(param.get_shape().as_list());
        nparam_vals += np.prod(param_shape);
    return nparam_vals;

def log_grads(cost_grads, cost_grad_vals, ind):
    cgv_ind = 0;
    nparams = len(cost_grads);
    for i in range(nparams):
        grad = cost_grads[i];
        grad_shape = grad.shape;
        ngrad_vals = np.prod(grad_shape);
        grad_reshape = np.reshape(grad, (ngrad_vals,));
        for ii in range(ngrad_vals):
            cost_grad_vals[ind, cgv_ind] = grad_reshape[ii];
            cgv_ind += 1;
    return None;

"""
def memory_extension(cost_grad_vals, array_cur_len):
    print('Doubling memory allocation for parameter logging.');
    #if (dynamics):
    #    As = np.concatenate((As, np.zeros((array_cur_len, As.shape[1], As.shape[2], As.shape[3]))), axis=0);
    #    sigma_epsilons = np.concatenate((sigma_epsilons, np.zeros((array_cur_len,sigma_epsilons.shape[1]))), axis=0);
    cost_grad_vals = np.concatenate((cost_grad_vals, np.zeros((array_cur_len, cost_grad_vals.shape[1]))), axis=0);
    return cost_grad_vals;
"""

def memory_extension(train_elbos, train_R2s, train_KLs, test_elbos, test_R2s, test_KLs, array_cur_len):
    print('Doubling memory allocation for parameter logging.');
    train_elbos = np.concatenate((train_elbos, np.zeros((array_cur_len, train_elbos.shape[1]))), axis=0);
    train_R2s = np.concatenate((train_R2s, np.zeros((array_cur_len, train_R2s.shape[1]))), axis=0);
    train_KLs = np.concatenate((train_KLs, np.zeros((array_cur_len, train_KLs.shape[1]))), axis=0);
    test_elbos = np.concatenate((test_elbos, np.zeros((array_cur_len, test_elbos.shape[1]))), axis=0);
    test_R2s = np.concatenate((test_R2s, np.zeros((array_cur_len, test_R2s.shape[1]))), axis=0);
    test_KLs = np.concatenate((test_KLs, np.zeros((array_cur_len, test_KLs.shape[1]))), axis=0);
    return train_elbos, train_R2s, train_KLs, test_elbos, test_R2s, test_KLs;
                

def autocovariance(X, tau_max, T, batch_size):
    # need to finish this
    X_toep = [];
    X_toep1 = [];
    X_toep2 = [];
    for i in range(tau_max+1):
        X_toep.append(X[:,:,i:((T-tau_max)+i)]);  # This will be (n x D x tau_max x (T- tau_max))
        X_toep1.append(X[:(batch_size//2),:,i:((T-tau_max)+i)]);
        X_toep2.append(X[(batch_size//2):,:,i:((T-tau_max)+i)]);

    X_toep = tf.reshape(tf.transpose(tf.convert_to_tensor(X_toep), [2, 0, 3, 1]), [D, tau_max+1, batch_size*(T-tau_max)]); # D x tau_max x (T-tau_max)*n
    X_toep1 = tf.reshape(tf.transpose(tf.convert_to_tensor(X_toep1), [2, 0, 3, 1]), [D, tau_max+1, (batch_size//2)*(T-tau_max)]); # D x tau_max x (T-tau_max)*n
    X_toep2 = tf.reshape(tf.transpose(tf.convert_to_tensor(X_toep2), [2, 0, 3, 1]), [D, tau_max+1, (batch_size//2)*(T-tau_max)]); # D x tau_max x (T-tau_max)*n
    
    X_toep_mc = X_toep - tf.expand_dims(tf.reduce_mean(X_toep, 2), 2);
    X_toep_mc1 = X_toep1 - tf.expand_dims(tf.reduce_mean(X_toep1, 2), 2);
    X_toep_mc2 = X_toep2 - tf.expand_dims(tf.reduce_mean(X_toep2, 2), 2);

    X_tau =  tf.cast((1/(batch_size*(T-tau_max))), tf.float64)*tf.matmul(X_toep_mc[:,:,:], tf.transpose(X_toep_mc, [0,2,1]))[:,:,0];
    X_tau1 =  tf.cast((1/((batch_size//2)*(T-tau_max))), tf.float64)*tf.matmul(X_toep_mc1[:,:,:], tf.transpose(X_toep_mc1, [0,2,1]))[:,:,0];
    X_tau2 =  tf.cast((1/((batch_size//2)*(T-tau_max))), tf.float64)*tf.matmul(X_toep_mc2[:,:,:], tf.transpose(X_toep_mc2, [0,2,1]))[:,:,0];

    X_tau_err = tf.reshape(X_tau - autocov_targ[:,:(tau_max+1)], (D*(tau_max+1),));
    X_tau_err1 = tf.reshape(X_tau1 - autocov_targ[:,:(tau_max+1)], (D*(tau_max+1),));
    X_tau_err2 = tf.reshape(X_tau2 - autocov_targ[:,:(tau_max+1)], (D*(tau_max+1),));
    tau_MSE = tf.reduce_sum(tf.square(X_tau_err));
    Tx_autocov = 0;
    Rx_autocov = 0;
    return Tx_autocov, Rx_autocov;


def adam_updates(params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999):
    ''' Adam optimizer '''
    updates = [];
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params);
    else:
        grads = cost_or_grads
    t = tf.Variable(1., 'adam_t')
    for p, g in zip(params, grads):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg', dtype=tf.float64)
        if mom1>0:
            v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v', dtype=tf.float64)
            v_t = mom1*v + (1. - mom1)*g
            v_hat = v_t / (1. - tf.pow(mom1,t))
            updates.append(v.assign(v_t))
        else:
            v_hat = g
        mg_t = mom2*mg + (1. - mom2)*tf.square(g)
        mg_hat = mg_t / (1. - tf.pow(mom2,t))
        g_t = v_hat / tf.sqrt(mg_hat + 1e-8)
        p_t = p - lr * g_t
        updates.append(mg.assign(mg_t))
        updates.append(p.assign(p_t))
    updates.append(t.assign_add(1))
    return tf.group(*updates)


def compute_VAR_cov_tf(As, Sigma_eps, D, K, T):
    # initialize the covariance matrix
    zcov = [[tf.eye(D)]];

    # compute the lagged noise-state covariance
    gamma = [Sigma_eps];
    for t in range(1,T):
        gamma_t = tf.zeros((D,D), dtype=tf.float64);
        for k in range(1, min(t,K)+1):
            gamma_t += tf.matmul(As[k-1], gamma[t-k]);
        gamma.append(gamma_t);

    # compute the off-block-diagonal covariance blocks
    # first row
    for s in range(1, T):
        zcov_0s = tf.zeros((D,D), dtype=tf.float64);
        for k in range(1, min(s, K)+1):
            tau = s-k;
            zcov_0tau = zcov[0][tau];
            zcov_0s += tf.matmul(zcov_0tau, tf.transpose(As[k-1]));
        zcov[0].append(zcov_0s);
        zcov.append([tf.transpose(zcov_0s)]);

    # remaining rows
    for t in range(1,T):
        for s in range(t, T):
            zcov_ts = tf.zeros((D,D), dtype=tf.float64);
            # compute the contribution of lagged state-state covariances
            for k_t in range(1, min(t,K)+1):
                tau_t = t-k_t;
                for k_s in range(1, min(s,K)+1):
                    tau_s = s-k_s;
                    zcov_tauttaus = zcov[tau_t][tau_s];
                    zcov_ts += tf.matmul(As[k_t-1], tf.matmul(zcov_tauttaus, tf.transpose(As[k_s-1])));
            # compute the contribution of lagged noise-state covariances
            if (t==s):
                zcov_ts += Sigma_eps;
            for k in range(1, min(s,K)+1):
                tau_s = s-k;
                if (tau_s >= t):
                    zcov_ts += tf.matmul(tf.transpose(gamma[tau_s-t]), tf.transpose(As[k-1]));

            zcov[t].append(zcov_ts);
            if (t != s):
                zcov[s].append(tf.transpose(zcov_ts));
                
    zcov = tf.convert_to_tensor(zcov);
    Zcov = tf.reshape(tf.transpose(zcov, [0,2,1,3]), (D*T, D*T));
    return Zcov;

def compute_VAR_cov_np(As, Sigma_eps, D, K, T):
    # Compute the analytic covariance of the VAR model

    # initialize the covariance matrix
    zcov = np.zeros((D*T, D*T));

    # compute the block-diagonal covariance
    zcov[:D, :D] = np.eye(D);

    # compute the lagged noise-state covariance
    gamma = [Sigma_eps];
    for t in range(1,T):
        gamma_t = np.zeros((D,D));
        for k in range(1, min(t,K)+1):
            gamma_t += np.dot(As[k-1], gamma[t-k]);
        gamma.append(gamma_t);

    # compute the off-block-diagonal covariance blocks
    # first row
    for s in range(1, T):
        zcov_0s = np.zeros((D,D));
        for k in range(1, min(s, K)+1):
            tau = s-k;
            zcov_0tau = zcov[:D, (D*tau):(D*(tau+1))]
            zcov_0s += np.dot(zcov_0tau, As[k-1].T);
        zcov[:D, (D*s):(D*(s+1))] = zcov_0s;
        zcov[(D*s):(D*(s+1)), :D] = zcov_0s.T;
            
    # remaining rows
    for t in range(1,T):
        for s in range(t, T):
            zcov_ts = np.zeros((D,D));
            # compute the contribution of lagged state-state covariances
            for k_t in range(1, min(t,K)+1):
                tau_t = t-k_t;
                for k_s in range(1, min(s,K)+1):
                    tau_s = s-k_s;
                    zcov_tauttaus = zcov[(D*tau_t):(D*(tau_t+1)), (D*tau_s):(D*(tau_s+1))]
                    zcov_ts += np.dot(As[k_t-1], np.dot(zcov_tauttaus, As[k_s-1].T));
                
            # compute the contribution of lagged noise-state covariances
            if (s==t):
                zcov_ts += Sigma_eps;
            for k in range(1, min(s,K)+1):
                tau_s = s-k;
                if (tau_s >= t):
                    zcov_ts += np.dot(gamma[tau_s-t].T, As[k-1].T);
            
            zcov[(D*t):(D*(t+1)), (D*s):(D*(s+1))] = zcov_ts;
            zcov[(D*s):(D*(s+1)), (D*t):(D*(t+1))] = zcov_ts.T;
    return zcov;


def simulate_VAR(As, Sigma_eps, T):
    K = As.shape[0];
    D = As.shape[1];
    mu = np.zeros((D,));
    z = np.zeros((D,T));
    z[:,0] = np.random.multivariate_normal(mu, np.eye(D));
    for t in range(1, T):
        Z_VAR_pred_t = np.zeros((D,));
        for k in range(K):
            if (t-(k+1) >= 0):
                Z_VAR_pred_t += np.dot(As[k], z[:, t-(k+1)]);
        eps_t = np.random.multivariate_normal(mu, Sigma_eps);
        z[:,t] = Z_VAR_pred_t + eps_t;
    return z;

def computeGPcov(l,T,eps):
    D = l.shape[0];
    for d in range(D):
        diffmat = np.zeros((T,T), dtype=np.float64);
        for i in range(T):
            for j in range(T):
                diffmat_ij = np.float64(i-j);
                diffmat[i,j] = diffmat_ij;
                if (i is not j):
                    diffmat[j,i] = diffmat_ij;
        GPcovd = np.exp(-np.square(diffmat) / (2.0*np.square(l[0]))) + eps*np.eye(T, dtype=np.float64);
        L = np.linalg.cholesky(GPcovd);
        if (d==0):
            GPcov = np.expand_dims(GPcovd, 0);
        else:
            GPcov = np.concatenate((GPcov, np.expand_dims(GPcovd,0)), axis=0);
    return GPcov;

def sampleGP(GPcov, n):
    D = GPcov.shape[0];
    T = GPcov.shape[1];
    for d in range(D):
        GPcovd = GPcov[d,:,:];
        L = np.linalg.cholesky(GPcovd);
        Z0d = np.dot(L, np.random.normal(np.zeros((T, n)), 1.0));
        if (d==0):
            Z_GP = np.expand_dims(Z0d, 0);
        else:
            Z_GP = np.concatenate((Z_GP, np.expand_dims(Z0d,0)), axis=0);
    return Z_GP


# Gabriel's MMD stuff
def MMD2u(K, m, n):
    """The MMD^2_u unbiased statistic.
    """
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    return 1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum()) + \
        1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum()) - \
        2.0 / (m * n) * Kxy.sum()
        
def compute_null_distribution(K, m, n, iterations=10000, verbose=False,
                              random_state=None, marker_interval=1000):
    """Compute the bootstrap null-distribution of MMD2u.
    """
    if type(random_state) == type(np.random.RandomState()):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        idx = rng.permutation(m+n)
        K_i = K[idx, idx[:, None]]
        mmd2u_null[i] = MMD2u(K_i, m, n)

    if verbose:
        print("")

    return mmd2u_null

def kernel_two_sample_test(X, Y, kernel_function='rbf', iterations=10000,
                           verbose=False, random_state=None, **kwargs):
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

    mmd2u_null = compute_null_distribution(K, m, n, iterations,
                                           verbose=verbose,
                                           random_state=random_state)
    p_value = max(1.0/iterations, (mmd2u_null > mmd2u).sum() /
                  float(iterations))
    if verbose:
        print("p-value ~= %s \t (resolution : %s)" % (p_value, 1.0/iterations))

    return mmd2u, mmd2u_null, p_value

def check_convergence(to_check, cur_ind, lag, thresh, criteria='lag_diff', wsize=500):
    len_to_check = len(to_check);
    vals = to_check[0];
    for i in range(1,len_to_check):
        vals = np.concatenate((vals, to_check[1]), axis=1);

    if (criteria=='lag_diff'):
        lag_mean = np.mean(vals[(cur_ind-(lag+wsize)):(cur_ind-lag),:], axis=0);
        cur_mean = np.mean(vals[(cur_ind-wsize):cur_ind,:], axis=0);
        log_param_diff = np.log(np.linalg.norm(lag_mean-cur_mean));
        has_converged = log_param_diff < thresh;
    elif (criteria=='grad_mean_ttest'):
        last_grads = vals[(cur_ind-lag):cur_ind, :];
        #Sigma_grads = np.dot(last_grads.T, last_grads) / (lag); # zero-mean covariance
        nvars = last_grads.shape[1];
        #mvt = mvd.MVT(np.zeros((nvars,)), Sigma_grads, lag);
        #grad_mean = np.mean(last_grads, 0);
        #t_cdf = mvt.cdf(grad_mean);
        #has_converged = (t_cdf > (thresh/2) and t_cdf < (1-(thresh/2)));
        #print('cdf val', t_cdf, 'convergence', has_converged);
        has_converged = True;
        for i in range(nvars):
            t, p = ttest_1samp(last_grads[:,i], 0);
            # if any grad mean is not zero, reject
            if (p < thresh):
                has_converged = False;
                break;
    return has_converged;

def test_convergence(mean_test_elbos, ind, wsize, delta_thresh):
    print('testing convergence');
    cur_mean_test_elbo = np.mean(mean_test_elbos[(ind-wsize+1):(ind+1)]);
    prev_mean_test_elbo = np.mean(mean_test_elbos[(ind-2*wsize+1):(ind-wsize+1)]);
    print('prev, cur');
    print(prev_mean_test_elbo, cur_mean_test_elbo);
    delta_elbo = (prev_mean_test_elbo - cur_mean_test_elbo) / prev_mean_test_elbo;
    print('delta elbo',delta_elbo);
    print('ret val', delta_elbo < delta_thresh)
    return delta_elbo < delta_thresh;

def find_convergence(mean_test_elbos, last_ind, wsize, delta_thresh):
    for ind in range(wsize, last_ind+1):
        if (test_convergence(mean_test_elbos, ind, wsize, delta_thresh)):
            return ind;
    return None;

def factors(n):
    return [f for f in range(1,n+1) if n%f==0]



