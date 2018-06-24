import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_kernels
from scipy.stats import ttest_1samp, multivariate_normal, dirichlet, invwishart, truncnorm
import statsmodels.sandbox.distributions.mv_normal as mvd
import matplotlib.pyplot as plt
from flows import LinearFlowLayer, PlanarFlowLayer, SimplexBijectionLayer, \
                  CholProdLayer, StructuredSpinnerLayer, TanhLayer, ExpLayer, \
                  SoftPlusLayer
import scipy.io as sio
import os
import re

p_eps = 10e-6;

def setup_IO(family, model_type_str, param_net_input_type, K, M, flow_dict, \
             param_net_hps, stochastic_eta, give_hint, random_seed):
    # set file I/O stuff
    resdir = 'results/MK/';
    eta_str = 'stochasticEta' if stochastic_eta else 'fixedEta';
    give_hint_str = 'giveHint_' if give_hint else '';
    flowstring = get_flowstring(flow_dict);
    if (model_type_str == 'EFN'):
        if (param_net_input_type == 'eta'):
            substr = '';
        elif (param_net_input_type == 'prior'):
            substr = 'a';
        elif (param_net_input_type == 'likelihood'):
            substr = 'b';
        elif (param_net_input_type == 'data'):
            substr = 'c';
        else:
            raise NotImplementedError();
        savedir = resdir + '/tb/' + 'EFN%s_%s_%s_%sD=%d_K=%d_M=%d_flow=%s_L=%d_rs=%d/' \
                  % (substr, family.name, eta_str, give_hint_str, family.D, K, M, flowstring, param_net_hps['L'], random_seed);
    else:
        savedir = resdir + '/tb/' + '%s_%s_D=%d_flow=%s_rs=%d/' % (model_type_str, family.name, family.D, flowstring, random_seed);
    return savedir

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
        layer_name, param_names, param_dims = layer.get_layer_info();
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
        layer_name, param_names, param_dims = layer.get_layer_info();
        nparams = len(param_names);
        layer_i_params = [];
        for j in range(nparams):
            if (j < 1):
                param_ij = tf.get_variable(layer_name+'_'+param_names[j], shape=param_dims[j], \
                                           dtype=tf.float64, \
                                           initializer=tf.glorot_uniform_initializer());
            elif (j==1):
                param_ij = tf.get_variable(layer_name+'_'+param_names[j], shape=param_dims[j], \
                                           dtype=tf.float64, \
                                           initializer=tf.glorot_uniform_initializer());
            elif (j==2):
                param_ij = tf.get_variable(layer_name+'_'+param_names[j], shape=param_dims[j], \
                                           dtype=tf.float64, \
                                           initializer=tf.glorot_uniform_initializer());
            layer_i_params.append(param_ij);
        theta.append(layer_i_params);
    return theta;

def count_layer_params(layer):
    num_params = 0;
    name, param_names, dims = layer.get_layer_info();
    nparams = len(dims);
    for j in range(nparams):
        num_params += np.prod(dims[j]);
    return num_params;

def construct_flow(flow_dict, D_Z, T):
    P = 0
    assert(P >= 0);
    flow_ids = flow_dict['flow_ids'];
    flow_repeats = flow_dict['flow_repeats'];
    nlayers = len(flow_ids);
    layer_ind = 1;
    layers = [];
    for i in range(nlayers):
        flow_id = flow_ids[i];
        num_repeats = flow_repeats[i];
        for j in range(num_repeats):
            if (flow_id == 'LinearFlowLayer'):
                layers.append(LinearFlowLayer('LinearFlow_Layer%d' % layer_ind, dim=D_Z));
            elif (flow_id == 'StructuredSpinnerLayer'):
                layers.append(StructuredSpinnerLayer('StructuredSpinner_Layer%d' % layer_ind, dim=D_Z));
            elif (flow_id == 'PlanarFlowLayer'):
                layers.append(PlanarFlowLayer('PlanarFlow_Layer%d' % layer_ind, dim=D_Z));
            elif (flow_id == 'TanhLayer'):
                layers.append(TanhLayer('Tanh_Layer%d' % layer_ind));
            layer_ind += 1;
    nlayers = len(layers); 

    num_theta_params = 0;
    for i in range(nlayers):
        layer = layers[i];
        num_theta_params += count_layer_params(layer);

    dynamics = P > 0;

    # Placeholder for initial condition of latent dynamical process
    if (dynamics):
        num_zi = 1;
    else:
        num_zi = T;

    Z0 = tf.placeholder(tf.float64, shape=(None, None, D_Z, num_zi), name='Z0');
    K = tf.shape(Z0)[0];
    M = tf.shape(Z0)[1];
    if (dynamics):
        # Note: The vector autoregressive parameters are vectorized for convenience in the 
        # unbiased gradient computation. 
        # Update, no need to vectorize. Fix this when implementing dynamics
        A_init = tf.random_normal((P*D_Z*D_Z,1), 0.0, 1.0, dtype=tf.float64);
        #A_init = np.array([[0.5], [0.0], [0.0], [0.5]], np.float64);
        log_sigma_eps_init = tf.abs(tf.random_normal((D_Z,1), 0.0, 0.1, dtype=tf.float64));
        noise_level = np.log(.02);
        log_sigma_eps_init = noise_level*np.array([[1.0], [1.0]], np.float64);
        A_vectorized = tf.get_variable('A_vectorized', dtype=tf.float64, initializer=A_init);
        A = tf.reshape(A_vectorized, [P,D_Z,D_Z]);
        log_sigma_eps = tf.get_variable('log_sigma_eps', dtype=tf.float64, initializer=log_sigma_eps_init);
        sigma_eps = tf.exp(log_sigma_eps);
        num_dyn_param_vals = P*D_Z*D_Z + D_Z;
        # contruct latent dynamics 
        Z_AR, base_log_p_z, Sigma_Z_AR = latent_dynamics(Z0, A, sigma_eps, T);
    else:
        # evaluate unit normal 
        assert(T==1); # TODO more care taken care for time series
        p0 = tf.expand_dims(tf.reduce_prod(tf.exp((-tf.square(Z0))/2.0)/np.sqrt(2.0*np.pi), axis=2), 2); 
        base_log_p_z = tf.log(p0[:,:,0,0]);
        num_dyn_param_vals = 0;
        Z_AR = Z0;
    return layers, Z0, Z_AR, base_log_p_z, P, num_zi, num_theta_params, num_dyn_param_vals;

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

def connect_flow(Z, layers, theta):
    Z_shape = tf.shape(Z);
    K = Z_shape[0];
    M = Z_shape[1];
    D_Z = Z_shape[2];
    T = Z_shape[3];

    sum_log_det_jacobians = tf.zeros((K,M), dtype=tf.float64);
    nlayers = len(layers);
    Z_by_layer = [];
    Z_by_layer.append(Z);
    for i in range(nlayers):
        layer = layers[i];
        theta_layer = theta[i];
        layer.connect_parameter_network(theta_layer);
        Z, sum_log_det_jacobians = layer.forward_and_jacobian(Z, sum_log_det_jacobians);
        Z_by_layer.append(Z);
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

def get_flowdict(GP_reg_layers, fully_connected_layers, planar_layers, spinner_layers, nonlin_spinner_layers):
    flow_ids = [];
    flow_repeats = [];
    if (GP_reg_layers):
        flow_ids.append('GP_EP_ConditionalRegressionLayer');
        flow_repeats.append(1); # no reason to have more than one here

    if (fully_connected_layers):
        flow_ids.append('LinearFlowLayer');
        flow_repeats.append(1); # no reason to have more than one here

    if (spinner_layers > 0):
        flow_ids.append('StructuredSpinnerLayer');
        flow_repeats.append(spinner_layers);

    for i in range(nonlin_spinner_layers):
        flow_ids.append('StructuredSpinnerLayer');
        flow_repeats.append(1);
        if not (i==(nonlin_spinner_layers-1)):
            flow_ids.append('TanhLayer');
            flow_repeats.append(1);

    if (planar_layers > 0):
        flow_ids.append('PlanarFlowLayer');
        flow_repeats.append(planar_layers);

    flow_dict = {'flow_ids':flow_ids, 'flow_repeats':flow_repeats};
    return flow_dict;

def print_flowdict(flow_dict):
    flow_ids = flow_dict['flow_ids'];
    flow_repeats = flow_dict['flow_repeats'];
    nlayers = len(flow_ids);
    for i in range(nlayers):
        print('%d %ss' % (flow_repeats[i], flow_ids[i]));
    return None;

def format_flowstring(flowstring):
    inds = [m.start() for m in re.finditer('1S_1T', flowstring)];
    ninds = len(inds);
    if (ninds == 0):
        return flowstring;

    diff_inds = np.diff(inds);
    start_inds = [];
    streak_counts = [];

    start_ind = inds[0];
    streak_count = 1;
    for i in range(ninds-1):
        if (diff_inds[i] == 6):
            streak_count += 1;
            if (i == ninds-2):
                start_inds.append(start_ind);
                streak_counts.append(streak_count);
        else:
            start_inds.append(start_ind);
            streak_counts.append(streak_count);
            if (i < ninds-2):
                start_ind = inds[i+1];
                streak_count = 1;

    keep_start_inds = [];
    keep_end_inds = [];
    keep_start_inds.append(0);
    for i in range(len(start_inds)):
        keep_end_inds.append(start_inds[i]);
        keep_start_inds.append(start_inds[i] + 6*streak_counts[i]);
    keep_end_inds.append(len(flowstring));

    flowstring_out = flowstring[keep_start_inds[0]:keep_end_inds[0]];
    for i in range(len(start_inds)):
        start_ind = start_inds[i];
        streak_count = streak_counts[i];
        flowstring_out += '%dST_' % streak_count;
        flowstring_out += flowstring[keep_start_inds[i+1]:keep_end_inds[i+1]];

    if (flowstring_out[-1] == '_'):
        flowstring_out = flowstring_out[:-1];
    print(flowstring_out);
    return flowstring_out;



def get_flowstring(flow_dict):
    flow_ids = flow_dict['flow_ids'];
    flow_repeats = flow_dict['flow_repeats'];
    num_flow_ids = len(flow_ids);
    flowidstring = '';
    for i in range(num_flow_ids):
        flowidstring += '%d%s' % (flow_repeats[i], flow_ids[i][0]);
        if (i < (num_flow_ids-1)):
            flowidstring += '_';
    return format_flowstring(flowidstring);

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

def memory_extension(cost_grad_vals, array_cur_len):
    print('Doubling memory allocation for parameter logging.');
    #if (dynamics):
    #    As = np.concatenate((As, np.zeros((array_cur_len, As.shape[1], As.shape[2], As.shape[3]))), axis=0);
    #    sigma_epsilons = np.concatenate((sigma_epsilons, np.zeros((array_cur_len,sigma_epsilons.shape[1]))), axis=0);
    cost_grad_vals = np.concatenate((cost_grad_vals, np.zeros((array_cur_len, cost_grad_vals.shape[1]))), axis=0);
    return cost_grad_vals;
                

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

def factors(n):
    return [f for f in range(1,n+1) if n%f==0]
