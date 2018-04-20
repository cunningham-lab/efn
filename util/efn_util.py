import tensorflow as tf
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_kernels
from scipy.stats import ttest_1samp, multivariate_normal, dirichlet, invwishart
import statsmodels.sandbox.distributions.mv_normal as mvd
import matplotlib.pyplot as plt
from flows import LinearFlowLayer, PlanarFlowLayer
import datetime
import os
from dirichlet import simplex

p_eps = 10e-6;
def setup_IO(exp_fam, D, flow_id, theta_nn_hps, stochastic_eta, random_seed):
# set file I/O stuff
    now = datetime.datetime.now();
    datestr = now.strftime("%Y-%m-%d_%H")
    resdir = 'results/' + datestr;
    if (not os.path.exists(resdir)):
        print('creating save directory:\n %s' % resdir);
        # This is an issue when processes are started in parallel.
        try:
            os.makedirs(resdir);
        except FileExistsError:
            print('%s already exists. Continuing.');

    eta_str = 'stochaticEta' if stochastic_eta else 'latticeEta';

    if ('L' in theta_nn_hps and 'upl' in theta_nn_hps):
        savedir = resdir + '/tb/' + 'EFN_%s_D=%d_%s_L=%d_upl=%d_RS=%d/' % (exp_fam, D, flow_id, theta_nn_hps['L'], theta_nn_hps['upl'], random_seed);
    else:
        savedir = resdir + '/tb/' + 'MEFN_%s_D=%d_%s_RS=%d/' % (exp_fam, D, flow_id, random_seed);
    return savedir

def construct_theta_network(eta, K_eta, flow_layers, theta_nn_hps):
    L_theta = theta_nn_hps['L']
    upl_theta = theta_nn_hps['upl'];
    L_flow = len(flow_layers);
    h = eta;
    for i in range(L_theta):
        with tf.variable_scope('ParamNetLayer%d' % (i+1)):
            h = tf.layers.dense(h, upl_theta, activation=tf.nn.tanh); # each layer will have ncons nodes (can change this)
    theta = [];
    for i in range(L_flow):
        layer = flow_layers[i];
        layer_name, param_names, param_dims = layer.get_layer_info();
        nparams = len(param_names);
        layer_i_params = [];
        # read each parameter out of the last layer.
        for j in range(nparams):
            num_elems = np.prod(param_dims[j]);
            A_shape = (upl_theta, num_elems);
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
    debug = False;
    L_flow = len(flow_layers);
    theta =[];
    for i in range(L_flow):
        layer = flow_layers[i];
        layer_name, param_names, param_dims = layer.get_layer_info();
        nparams = len(param_names);
        layer_i_params = [];
        if (debug and i==0):
            u_init = np.expand_dims(np.array([.78, 1.3]), 1);
            w_init = np.expand_dims(np.array([-.15, -1.28]), 1);
            b_init = .6*np.ones((1,1), np.float64);
            param_u = tf.get_variable(layer_name+'_'+param_names[0], \
                                           dtype=tf.float64, \
                                           initializer=u_init);
            param_w = tf.get_variable(layer_name+'_'+param_names[1], \
                                           dtype=tf.float64, \
                                           initializer=w_init);
            param_b = tf.get_variable(layer_name+'_'+param_names[2], \
                                           dtype=tf.float64, \
                                           initializer=b_init);
            layer_i_params = [param_u, param_w, param_b];
        else:
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

def construct_flow(flow_id, D_Z, T):
    datadir = 'flows/';
    fname = datadir + '%s.npz' % flow_id;
    flowfile = np.load(fname);
    num_linear_lyrs = int(flowfile['num_linear_layers']);
    num_planar_lyrs = int(flowfile['num_planar_layers']);
    P = int(flowfile['P']); # order of VAR dynamics
    assert(P >= 0);
    layers = [];
    for i in range(num_linear_lyrs):
        layers.append(LinearFlowLayer('LinearFlow%d' % (i+1), dim=D_Z));
    for i in range(num_planar_lyrs):
        layers.append(PlanarFlowLayer('PlanarFlow%d' % (i+1), dim=D_Z));
    nlayers = len(layers); 

    dynamics = P > 0;

    # Placeholder for initial condition of latent dynamical process
    if (dynamics):
        num_zi = 1;
    else:
        num_zi = T;

    Z0 = tf.placeholder(tf.float64, shape=(None, None, D_Z, num_zi));
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

    return layers, Z0, Z_AR, base_log_p_z, P, num_zi, num_dyn_param_vals;

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

def connect_flow(Z, layers, theta, exp_fam, K_eta, M_eta):
    Z_shape = tf.shape(Z);
    D = Z_shape[2];
    T = Z_shape[3];

    sum_log_det_jacobians = tf.zeros((K_eta,M_eta), dtype=tf.float64);
    nlayers = len(layers);
    input_to_log_abs_list = [];
    log_det_jac_list = [];
    for i in range(nlayers):
        layer = layers[i];
        theta_layer = theta[i];
        layer.connect_parameter_network(theta_layer);
        Z, sum_log_det_jacobians, log_det_jac, input_to_log_abs = layer.forward_and_jacobian(Z, sum_log_det_jacobians);
        input_to_log_abs_list.append(input_to_log_abs);
        log_det_jac_list.append(log_det_jac);
    Z_pf = Z;
    # final layer translates to the support
    if (exp_fam == 'dirichlet'):
        # TODO need to redo this
        ex = tf.exp(Z);
        den = tf.reduce_sum(ex, 2) + 1.0;
        Z_simplex = tf.concat((ex / tf.expand_dims(den, 2), 1.0 / tf.expand_dims(den, 2)), axis=2);
        log_dets = tf.log(1.0 - (tf.reduce_sum(ex, 2) / den)) - tf.cast(D, tf.float64)*tf.log(den) + tf.reduce_sum(Z, 2);
        #Z = tf.exp(Z) / tf.expand_dims((tf.reduce_sum(tf.exp(Z) ,axis=2) + 1), 2); 
        # compute the jacobian using matrix determinant lemma
        #u = Z;
        #Adiag = u;
        #Ainvdiag = 1.0 / u;
        #v = -u;
        #g_det_jacobian = tf.multiply((1.0+tf.reduce_sum(tf.multiply(tf.multiply(v,Ainvdiag), u), axis=2)), tf.reduce_prod(Adiag, axis=2));
        #left_term = (1.0+tf.reduce_sum(tf.multiply(tf.multiply(v,Ainvdiag), u), axis=2));
        #detA = tf.reduce_prod(Adiag, axis=2);
        #g_log_det_jacobian = tf.log(g_det_jacobian);
        #g_log_det_jacobian = tf.reduce_sum(g_log_det_jacobian, 2); # sum across the temporal dimension
        #g_det_jacobian = tf.multiply(tf.reduce_prod(Z, 2), (1-tf.reduce_sum(Z, 2)));
        #g_log_det_jacobian = tf.log(tf.abs(g_det_jacobian[:,:,0]));
        sum_log_det_jacobians += log_dets[:,:,0];
        #Z = tf.concat((Z, tf.expand_dims(1-tf.reduce_sum(Z, axis=2), 2)), axis=2);

    return Z_simplex, sum_log_det_jacobians, Z_pf, log_det_jac_list, input_to_log_abs_list


def approxKL(z_i, y_k, X_k, exp_fam, params, plot=False):
    log_Q = y_k[:,0];
    #log_Q0 = y0_k[:,0];
    Q = np.exp(log_Q);
    #Q0 = np.exp(log_Q0);
    if (exp_fam == 'normal'):
        mu = params['mu'];
        Sigma = params['Sigma'];
        dist = multivariate_normal(mean=mu, cov=Sigma);
        log_P = dist.logpdf(X_k);
        KL = np.mean(log_Q - log_P);
        if (plot):
            batch_size = X_k.shape[0];
            X_true = np.random.multivariate_normal(mu, Sigma, (batch_size,));
            log_P_true = dist.logpdf(X_true);
            xmin = min(np.min(X_k[:,0]), np.min(X_true[:,0]));
            xmax = max(np.max(X_k[:,0]), np.max(X_true[:,0]));
            ymin = min(np.min(X_k[:,1]), np.min(X_true[:,1]));
            ymax = max(np.max(X_k[:,1]), np.max(X_true[:,1]));
            fig = plt.figure(figsize=(8, 4));
            fig.add_subplot(1,2,1);
            plt.scatter(X_k[:,0], X_k[:,1], c=log_Q);
            plt.xlim([xmin, xmax]);
            plt.ylim([ymin, ymax]);
            plt.colorbar();
            plt.title('mu: [%.1f, %.1f] Sigma: [%.2f, %.2f, %.2f]' % (mu[0], mu[1], Sigma[0,0], Sigma[0,1], Sigma[1,1]));
            fig.add_subplot(1,2,2);
            plt.scatter(X_true[:,0], X_true[:,1], c=log_P_true);
            plt.xlim([xmin, xmax]);
            plt.ylim([ymin, ymax]);
            plt.colorbar();
            plt.show();
    elif (exp_fam == 'dirichlet'):
        alpha = params['alpha'];
        dist = dirichlet(np.float64(alpha));
        # For higher-D dirichlet's the numpy sum can be non-one by a bit or two
        # relative to the tensorflow sum.  Must have different hardware optims.
        # We just do this extra normalization step in 64 bit for the logpdf eval.
        batch_size = X_k.shape[0];
        X_k = np.float64(X_k);
        X_k = X_k / np.expand_dims(np.sum(X_k, 1), 1);
        log_P = dist.logpdf(X_k.T);
        #print('******** KL **********');
        print(log_Q.shape, log_P.shape);
        print(np.mean(log_Q), np.mean(log_P));
        n = 100;
        KLs = np.zeros((n,));
        n_i = batch_size // 10;
        inds = np.arange(batch_size);
        for i in range(n):
            inds_i = np.random.choice(inds, n_i);
            log_Q_i = log_Q[inds_i];
            X_i = X_k[inds_i, :];
            log_P_i = dist.logpdf(X_i.T);
            KLs[i] = np.mean(log_Q_i - log_P_i);
        KL = np.mean(log_Q - log_P);
        print(KL);
        #print(KL);
        if (plot):
            log_diff = log_Q - log_P;
            minval = min([np.min(log_Q), np.min(log_P), np.min(log_diff)]);
            maxval = max([np.max(log_Q), np.max(log_P), np.max(log_diff)]);
            batch_size = X_k.shape[0];
            #X_true = np.random.dirichlet(alpha, (batch_size,));
            #log_P_true = dist.logpdf(X_true.T);
            fig = plt.figure(figsize=(8, 8));
            fig.add_subplot(2,2,1);
            simplex.scatter(X_k, connect=False, c=log_Q, vmin=minval, vmax=maxval);
            plt.colorbar();
            plt.title('colored by log Q(z | theta, eta)');

            fig.add_subplot(2,2,2);
            pts = simplex.scatter(X_k, connect=False, c=log_P, vmin=minval, vmax=maxval);
            plt.colorbar();
            plt.title('colored by log P_true(z | eta)');

            fig.add_subplot(2,2,3);
            pts = simplex.scatter(X_k, connect=False, c=log_diff, vmin=minval, vmax=maxval);
            plt.colorbar();
            plt.title('colored by log Q - log P_true');

            fig.add_subplot(2,2,4);
            pts = plt.hist(KLs);
            plt.title('hist of resampled KLs');
            plt.show();
            """
            buf = .2;
            xmin = min(np.min(z_i[:,0]), np.min(z_i[:,0])) - buf;
            xmax = max(np.max(z_i[:,0]), np.max(z_i[:,0])) + buf;
            ymin = min(np.min(z_i[:,1]), np.min(z_i[:,1])) - buf;
            ymax = max(np.max(z_i[:,1]), np.max(z_i[:,1])) + buf;
            fig.add_subplot(2,2,1);
            print(z_i.shape, log_Q.shape);
            plt.scatter(z_i[:,0], z_i[:,1], c=log_Q);
            plt.colorbar();
            plt.xlim([xmin, xmax]);
            plt.ylim([ymin, ymax]);

            fig.add_subplot(2,2,2);
            plt.scatter(z_i[:,0], z_i[:,1], c=log_P);
            plt.colorbar();
            plt.xlim([xmin, xmax]);
            plt.ylim([ymin, ymax]);
            plt.show();
            """
    return KL;

def checkH(y_k, exp_fam, params):
    log_Q = y_k[:,0];
    H = np.mean(-log_Q);
    if (exp_fam == 'normal'):
        mu = params['mu'];
        Sigma = params['Sigma'];
        dist = multivariate_normal(mean=mu, cov=Sigma);
        H_true = dist.entropy();
    if (exp_fam == 'dirichlet'):
        alpha = params['alpha'];
        dist = dirichlet(alpha);
        H_true = dist.entropy();
    print('H = %.3f/%.3f' % (H, H_true));
    return None;

def computeMoments(X, exp_fam, D, T):
    X_shape = tf.shape(X);
    K = X_shape[0];
    M = X_shape[1];
    if (T==1):
        if (exp_fam == 'normal'):
            cov_con_mask = np.triu(np.ones((D,D), dtype=np.bool_), 0);
            X_flat = tf.reshape(tf.transpose(X, [0, 1, 3, 2]), [K, M, D]); # samps x D
            Tx_mean = X_flat;
            XXT = tf.matmul(tf.expand_dims(X_flat, 3), tf.expand_dims(X_flat, 2));

            #Tx_cov = tf.transpose(tf.boolean_mask(tf.transpose(X_cov, [1,2,0]), _cov_con_mask)); # [n x (D*(D-1)/2 )]
            Tx_cov = tf.reshape(XXT, [K,M, D*D]);
            Tx = tf.concat((Tx_mean, Tx_cov), axis=2);
        elif (exp_fam == 'dirichlet'):
            X_flat = tf.reshape(tf.transpose(X, [0, 1, 3, 2]), [K, M, D]); # samps x D
            Tx_log = tf.log(X_flat);
            Tx = Tx_log;
        else: 
            raise NotImplementedError;
    else:
        raise NotImplementedError; 
    return Tx;

def computeLogBaseMeasure(X, exp_fam, D, T):
    X_shape = tf.shape(X);
    K = X_shape[0];
    M = X_shape[1];
    if (T==1): # only handling this for T=1 atm
        if (exp_fam == 'normal'):
            Bx = tf.zeros((K,M), dtype=tf.float64);
        elif (exp_fam == 'dirichlet'):
            Bx = tf.log(tf.divide(1.0, tf.reduce_prod(X, [2])));
            Bx = Bx[:,:,0]; # remove singleton time dimension
        else:
            raise NotImplementedError;
    else:
        raise NotImplementedError;
    return Bx;

def cost_fn(eta, log_p_zs, Tx, Bx, K_eta, cost_type):
    y = log_p_zs
    cost = 0.0;
    R2s = [];
    for k in range(K_eta):
        # get eta-specific log-probs and T(x)'s
        y_k = tf.expand_dims(y[k,:], 1);
        Tx_k = Tx[k,:,:];
        Bx_k = tf.expand_dims(Bx[k,:], 1);
        eta_k = tf.expand_dims(eta[k,:], 1);
        # compute optimial linear regression offset term for eta
        alpha_k = tf.reduce_mean(y_k - (tf.matmul(Tx_k, eta_k) + Bx_k));
        residuals = y_k - (tf.matmul(Tx_k, eta_k)+Bx_k) - alpha_k;
        RSS_k = tf.matmul(tf.transpose(residuals), residuals);
        y_k_mc = y_k - tf.reduce_mean(y_k);
        TSS_k = tf.reduce_sum(tf.square(y_k_mc));
        # compute the R^2 of the exponential family fit
        R2s.append(1.0 - (RSS_k[0,0] / TSS_k));
        if (cost_type == 'reg'):
            cost += RSS_k[0,0];
        elif (cost_type == 'KL'):
            cost += tf.reduce_mean(y_k - (tf.matmul(Tx_k, eta_k) + Bx_k));

    return cost, R2s;

def getEtas(constraint_id, K_eta):
    D_Z, K_eta, params, constraint_type = load_constraint_info(constraint_id);
    datadir = 'constraints/'
    fname = datadir + '%s.npz' % constraint_id;
    confile = np.load(fname);
    if (constraint_type == 'normal'):
        mu_targs = confile['mu_targs'];
        Sigma_targs = confile['Sigma_targs'];
        etas = [];
        for k in range(K_eta):
            eta = normal_eta(mu_targs[k,:], Sigma_targs[k,:,:]);
            etas.append(eta);

        mu_OL_targs = confile['mu_OL_targs'];
        Sigma_OL_targs = confile['Sigma_OL_targs'];
        off_lattice_etas = [];
        for k in range(K_eta):
            ol_eta1 = np.float64(np.dot(np.linalg.inv(Sigma_OL_targs[k,:,:]), np.expand_dims(mu_OL_targs[k,:], 2)));
            ol_eta2 = np.float64(-np.linalg.inv(Sigma_OL_targs[k,:,:]) / 2);
            ol_eta = np.concatenate((ol_eta1, np.reshape(ol_eta2, [D_Z*D_Z, 1])), 0);
            off_lattice_etas.append(ol_eta);


    elif (constraint_type == 'dirichlet'):
        alpha_targs = np.float64(confile['alpha_targs']);
        etas = [];
        for k in range(K_eta):
            eta = np.expand_dims(alpha_targs[k,:], 1);
            etas.append(eta);

        alpha_OL_targs = np.float64(confile['alpha_OL_targs']);
        off_lattice_etas = [];
        for k in range(K_eta):
            ol_eta = np.expand_dims(alpha_OL_targs[k,:], 1);
            off_lattice_etas.append(ol_eta);
    return etas, off_lattice_etas;

def normal_eta(mu, Sigma):
    D_Z = mu.shape[0];
    eta1 = np.float64(np.dot(np.linalg.inv(Sigma), np.expand_dims(mu, 2)));
    eta2 = np.float64(-np.linalg.inv(Sigma) / 2);
    eta = np.concatenate((eta1, np.reshape(eta2, [D_Z*D_Z, 1])), 0);
    return eta;


def drawEtas(exp_fam, D_Z, K_eta, n_k):
    if (exp_fam == 'normal'):
        mu_targs = np.zeros((K_eta, D_Z));
        Sigma_targs = np.zeros((K_eta, D_Z, D_Z));
        ncons = D_Z+D_Z**2;
        eta = np.zeros((K_eta, ncons));
        df_fac = 2;
        df = df_fac*D_Z;
        Sigma_dist = invwishart(df=df, scale=df_fac*np.eye(D_Z));
        for k in range(K_eta):
            mu_k = np.random.multivariate_normal(np.zeros((D_Z,)), np.eye(D_Z));
            Sigma_k = Sigma_dist.rvs(1);
            eta_k = normal_eta(mu_k, Sigma_k);
            eta[k,:] = eta_k[:,0];
            mu_targs[k,:] = mu_k;
            Sigma_targs[k,:,:] = Sigma_k;
        params = {'mu_targs':mu_targs, 'Sigma_targs':Sigma_targs};
    if (exp_fam == 'dirichlet'):
        D_X = D_Z + 1;
        eta = np.zeros((K_eta, D_X));
        alpha_targs = np.zeros((K_eta, D_X));
        for k in range(K_eta):
            alpha_k = np.random.uniform(1, 3, (D_X,));
            eta_k = alpha_k;
            eta[k,:] = eta_k;
            alpha_targs[k,:] = alpha_k;
        params = {'alpha_targs':alpha_targs};
    return eta, params;

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

def plotMefnTraining(exp_fam, R2s, KLs, X, log_P, params, check_rate, iters, titlestr):
    fontsize = 14;
    num_checks = iters // check_rate;
    print('R2s shape', R2s.shape);
    K_eta = R2s.shape[1];
    its = check_rate*np.arange(1,num_checks+1, dtype=float);
    print(its);
    its = np.tile(np.expand_dims(its, 1), [1,K_eta]);
    its_vec = np.reshape(its, (num_checks*K_eta,));
    R2s_vec = np.reshape(R2s[:num_checks, :], (num_checks*K_eta,))
    KLs_vec = np.reshape(KLs[:num_checks, :], (num_checks*K_eta,))
    size = np.ones((num_checks*K_eta,));
    fig = plt.figure(figsize=(6,5));
        

    fig.add_subplot(2,2,1);
    plt.plot([np.min(its), np.max(its)], [0,0], 'tab:gray');
    plt.scatter(its_vec, R2s_vec, size,c='k');
    plt.legend(['goal', 'model']);
    plt.xlabel('iterations', fontsize=fontsize);
    plt.xlabel('iterations', fontsize=fontsize);
    plt.ylabel('R^2$', fontsize=fontsize)
    plt.ylim([-.2, 1]);

    fig.add_subplot(2,2,3);
    plt.plot([np.min(its), np.max(its)], [0,0], 'tab:gray');
    plt.scatter(its_vec, KLs_vec, size,c='k');
    plt.legend(['goal', 'model']);
    plt.xlabel('iterations', fontsize=fontsize);
    plt.ylabel('KL', fontsize=fontsize)

    # plot the distribution
    if (exp_fam == 'dirichlet'):
        fig.add_subplot(2,2,2);
        batch_size = X.shape[0];
        alpha = params['alpha'];
        D = alpha.shape[0];
        if (D == 3):
            X_true = np.random.dirichlet(alpha, (batch_size,));
            dist = dirichlet(np.float64(alpha));
            log_P_true = dist.logpdf(X_true.T);
            simplex.scatter(X, connect=False, c=log_P);
            plt.colorbar();
            plt.title('Q(X | theta, eta)', fontsize=fontsize)

            fig.add_subplot(2,2,4);
            simplex.scatter(X_true, connect=False, c=log_P_true);
            plt.colorbar();
            plt.title('true P(X | eta)');
            plt.suptitle(titlestr, fontsize=fontsize+2);
    return fig;
    

def factors(n):
    return [f for f in range(1,n+1) if n%f==0]
