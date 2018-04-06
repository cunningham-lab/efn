import tensorflow as tf
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_kernels
from scipy.stats import ttest_1samp, multivariate_normal
import statsmodels.sandbox.distributions.mv_normal as mvd
import matplotlib.pyplot as plt
from flows import LinearFlowLayer, PlanarFlowLayer

p_eps = 10e-6;
def initialize_optimization_parameters(sess, optimizer, all_params):
    nparams = len(all_params);
    slot_names = optimizer.get_slot_names();
    num_slot_names = len(slot_names);
    for i in range(nparams):
        param = all_params[i];
        for j in range(num_slot_names):
            slot_name = slot_names[j];
            opt_var = optimizer.get_slot(param, slot_name);
            sess.run(opt_var.initializer);
    # Adam optimizer is the only tf optimizer that has this slot variable issue
    # href https://github.com/tensorflow/tensorflow/issues/8057
    if (isinstance(optimizer, tf.train.AdamOptimizer)):
        sess.run(optimizer._beta1_power.initializer);
        sess.run(optimizer._beta2_power.initializer);
    return None;

def load_constraint_info(constraint_id):
    datadir = 'constraints/';
    fname = datadir + '%s.npz' % constraint_id;
    confile = np.load(fname);
    if (constraint_id[:6] == 'normal'):
        constraint_type = 'normal';
    elif (constraint_id[:9] == 'dirichlet'):
        constraint_type = 'dirichlet';
    if (constraint_type == 'normal'):
        mu_targs = confile['mu_targs'];
        mu_OL_targs = confile['mu_OL_targs'];
        Sigma_targs = confile['Sigma_targs'];
        Sigma_OL_targs = confile['Sigma_OL_targs'];
        params = {'mu_targs':mu_targs, 'mu_OL_targs':mu_OL_targs, \
                  'Sigma_targs':Sigma_targs, 'Sigma_OL_targs':Sigma_OL_targs};
        K_eta, D = mu_targs.shape;
    elif (constraint_type == 'dirichlet'):
        alpha_targs = confile['alpha_targs'];
        alpha_OL_targs = confile['alpha_targs'];
        params = {'alpha_targs':alpha_targs, 'alpha_OL_targs':alpha_OL_targs};
        K_eta, D_X = alpha_targs.shape;
        D = D_X-1;
    return D, K_eta, params, constraint_type;

def load_constraint_lims(constraint_id):
    datadir = 'constraints/';
    fname = datadir + '%s.npz' % constraint_id;
    confile = np.load(fname);
    if (constraint_id[:6] == 'normal'):
        constraint_type = 'normal';
    if (constraint_type == 'normal'):
        mean_lims = confile['mean_lims'];
        var_lims = confile['var_lims'];
        cor_lims = confile['cor_lims'];
        param_lims = {'mean_lims':mean_lims, 'var_lims':var_lims, \
                      'cor_lims':cor_lims};
        D_Z =2;
        ncons = D_Z + D_Z*D_Z;
    return param_lims, ncons, constraint_type;

def construct_flow(flow_id, D_Z):
    datadir = 'flows/';
    fname = datadir + '%s.npz' % flow_id;
    flowfile = np.load(fname);
    num_linear_lyrs = int(flowfile['num_linear_layers']);
    num_planar_lyrs = int(flowfile['num_planar_layers']);
    K = int(flowfile['K']);
    layers = [];
    for i in range(num_linear_lyrs):
        layers.append(LinearFlowLayer('LinearFlow%d' % (i+1), dim=D_Z));
    for i in range(num_planar_lyrs):
        layers.append(PlanarFlowLayer('PlanarFlow%d' % (i+1), dim=D_Z));
    nlayers = len(layers); 

    return layers, num_linear_lyrs, num_planar_lyrs, K;


def approxKL(y_k, X_k, constraint_type, params, plot=False):
    log_Q = y_k[:,0];
    Q = np.exp(log_Q);
    if (constraint_type == 'normal'):
        mu = params['mu'];
        Sigma = params['Sigma'];
        dist = multivariate_normal(mean=mu, cov=Sigma);
        log_P = dist.logpdf(X_k);
        KL = np.sum(np.multiply(Q, log_Q - log_P));
        if (plot):
            n = X_k.shape[0];
            sizes = 40*np.ones((n,1));
            minval = min(np.min(log_Q), np.min(log_P));
            maxval = max(np.max(log_Q), np.max(log_P));
            print(KL);
            fig = plt.figure(figsize=(12,4));
            fig.add_subplot(1,3,1);
            plt.scatter(X_k[:,0], X_k[:,1], sizes, log_Q, vmin=minval, vmax=maxval);
            plt.colorbar();
            plt.title('Sigma: [%.3f, %.3f, %.3f], logQ' % (Sigma[0,0], Sigma[0,1], Sigma[1,1]));
            fig.add_subplot(1,3,2);
            plt.scatter(X_k[:,0], X_k[:,1], sizes, log_P);
            plt.title('log P');
            plt.colorbar();
            fig.add_subplot(1,3,3);
            plt.scatter(X_k[:,0], X_k[:,1], sizes, log_Q-log_P);
            plt.title('logQ - logP');
            plt.colorbar();
            plt.show();
    if (constraint_type == 'dirichlet'):
        raise NotImplementedError('dirichlet KL');
        

    return KL;


def computeMoments(X, constraint_id):
    X_shape = tf.shape(X);
    batch_size = X_shape[0];
    D_Z, K_eta, params, constraint_type = load_constraint_info(constraint_id);
    T = X_shape[2];
    if (constraint_type == 'normal'):
        D_X = D_Z;
        cov_con_mask = np.triu(np.ones((D_X,D_X), dtype=np.bool_), 0);
        X_flat = tf.reshape(tf.transpose(X, [0, 2, 1]), [T*batch_size, D_X]); # samps x D
        Tx_mean = X_flat;
        XXT = tf.matmul(tf.expand_dims(X_flat, 2), tf.expand_dims(X_flat, 1));

        #Tx_cov = tf.transpose(tf.boolean_mask(tf.transpose(X_cov, [1,2,0]), _cov_con_mask)); # [n x (D*(D-1)/2 )]
        Tx_cov = tf.reshape(XXT, [T*batch_size, D_X*D_X]);
        Tx = tf.concat((Tx_mean, Tx_cov), axis=1);
    elif (constraint_type == 'dirichlet'):
        D_X = D_Z + 1;
        X_flat = tf.reshape(tf.transpose(X, [0, 2, 1]), [T*batch_size, D_X]); # samps x D
        Tx_log = tf.log(X_flat);
        Tx = Tx_log;
    else:
        raise NotImplementedError;


    return Tx;

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
            ol_eta1 = np.float32(np.dot(np.linalg.inv(Sigma_OL_targs[k,:,:]), np.expand_dims(mu_OL_targs[k,:], 2)));
            ol_eta2 = np.float32(-np.linalg.inv(Sigma_OL_targs[k,:,:]) / 2);
            ol_eta = np.concatenate((ol_eta1, np.reshape(ol_eta2, [D_Z*D_Z, 1])), 0);
            off_lattice_etas.append(ol_eta);


    elif (constraint_type == 'dirichlet'):
        alpha_targs = np.float32(confile['alpha_targs']);
        etas = [];
        for k in range(K_eta):
            eta = np.expand_dims(alpha_targs[k,:], 1);
            etas.append(eta);

        alpha_OL_targs = np.float32(confile['alpha_OL_targs']);
        off_lattice_etas = [];
        for k in range(K_eta):
            ol_eta = np.expand_dims(alpha_OL_targs[k,:], 1);
            off_lattice_etas.append(ol_eta);
    return etas, off_lattice_etas;

def normal_eta(mu, Sigma):
    D_Z = mu.shape[0];
    eta1 = np.float32(np.dot(np.linalg.inv(Sigma), np.expand_dims(mu, 2)));
    eta2 = np.float32(-np.linalg.inv(Sigma) / 2);
    eta = np.concatenate((eta1, np.reshape(eta2, [D_Z*D_Z, 1])), 0);
    return eta;


def drawEtas(constraint_id, K_eta, n_k):
    param_lims, ncons, constraint_type = load_constraint_lims(constraint_id);
    n = K_eta * n_k;
    if (constraint_type == 'normal'):
        D_Z = 2;
        mu_targs = np.zeros((K_eta, D_Z));
        Sigma_targs = np.zeros((K_eta, D_Z, D_Z));

        mean_lims = param_lims['mean_lims'];
        var_lims = param_lims['var_lims'];
        cor_lims = param_lims['cor_lims'];
        eta = np.zeros((n, ncons));
        for k in range(K_eta):
            k_start = k*n_k;
            mu_k = np.random.uniform(mean_lims[0], mean_lims[1], (D_Z,));
            vars_k = np.random.uniform(var_lims[0], var_lims[1], (D_Z,));
            corr_k = np.random.uniform(cor_lims[0], cor_lims[1]);
            offdiag_k = corr_k*np.sqrt(vars_k[0])*np.sqrt(vars_k[1]);
            Sigma_k = np.array([[vars_k[0], offdiag_k], [offdiag_k, vars_k[1]]]);
            eta_k = normal_eta(mu_k, Sigma_k);
            for i in range(n_k):
                eta[k_start+i,:] = eta_k[:,0];
            mu_targs[k,:] = mu_k;
            Sigma_targs[k,:,:] = Sigma_k;
    params = {'mu_targs':mu_targs, 'Sigma_targs':Sigma_targs};
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

    X_tau =  tf.cast((1/(batch_size*(T-tau_max))), tf.float32)*tf.matmul(X_toep_mc[:,:,:], tf.transpose(X_toep_mc, [0,2,1]))[:,:,0];
    X_tau1 =  tf.cast((1/((batch_size//2)*(T-tau_max))), tf.float32)*tf.matmul(X_toep_mc1[:,:,:], tf.transpose(X_toep_mc1, [0,2,1]))[:,:,0];
    X_tau2 =  tf.cast((1/((batch_size//2)*(T-tau_max))), tf.float32)*tf.matmul(X_toep_mc2[:,:,:], tf.transpose(X_toep_mc2, [0,2,1]))[:,:,0];

    X_tau_err = tf.reshape(X_tau - autocov_targ[:,:(tau_max+1)], (D*(tau_max+1),));
    X_tau_err1 = tf.reshape(X_tau1 - autocov_targ[:,:(tau_max+1)], (D*(tau_max+1),));
    X_tau_err2 = tf.reshape(X_tau2 - autocov_targ[:,:(tau_max+1)], (D*(tau_max+1),));
    tau_MSE = tf.reduce_sum(tf.square(X_tau_err));
    Tx_autocov = 0;
    Rx_autocov = 0;
    return Tx_autocov, Rx_autocov;

def latent_dynamics(Z0, A, sigma_eps, T):
    k = A.shape[0];
    D = tf.shape(A)[1];
    batch_size = tf.shape(Z0)[0];
    Z_AR = Z0;
    for t in range(1, T):
        Z_AR_pred_t = tf.zeros((batch_size, D));
        for i in range(k):
            if (t-(i+1) >= 0):
                Z_AR_pred_t += tf.transpose(tf.matmul(A[i,:,:], tf.transpose(Z_AR[:,:,t-(i+1)])));
        epsilon = tf.transpose(tf.multiply(sigma_eps, tf.random_normal((D, batch_size),0,1)));
        Z_AR_t = Z_AR_pred_t + epsilon;

        Z_AR = tf.concat((Z_AR, tf.expand_dims(Z_AR_t, 2)), axis=2);

    Sigma_eps = tf.diag(tf.square(sigma_eps[:,0]));
    Sigma_Z_AR = compute_VAR_cov_tf(A, Sigma_eps, D, k, T);
    Sigma_Z_AR = Sigma_Z_AR;
    
    Z_AR_dist = tf.contrib.distributions.MultivariateNormalFullCovariance(tf.zeros((D*T,)), Sigma_Z_AR);

    Z_AR_dist_shaped = tf.reshape(tf.transpose(Z_AR, [0, 2, 1]), (batch_size, D*T));
    log_p_Z_AR = Z_AR_dist.log_prob(Z_AR_dist_shaped);
    return Z_AR, log_p_Z_AR, Sigma_Z_AR;


def time_invariant_flow(Z, layers, theta, constraint_type):
    Z_shape = tf.shape(Z);
    batch_size = Z_shape[0];
    D = Z_shape[1];
    T = Z_shape[2];

    sum_log_det_jacobians = 0.0;
    nlayers = len(layers);
    for i in range(nlayers):
        layer = layers[i];
        theta_layer = theta[i];
        layer.connect_parameter_network(theta_layer);
        Z, sum_log_det_jacobians = layer.forward_and_jacobian(Z, sum_log_det_jacobians);

    # final layer translates to the support
    if (constraint_type == 'dirichlet'):
        # TODO this needs to be compatable with current shaping of Z
        Z = tf.exp(Z) / tf.expand_dims((tf.reduce_sum(tf.exp(Z) ,axis=1) + 1), 1); 
        # compute the jacobian using matrix determinant lemma
        u = Z;
        Adiag = u;
        Ainvdiag = 1.0 / u;
        v = -u;
        g_det_jacobian = tf.multiply((1.0+tf.reduce_sum(tf.multiply(tf.multiply(v,Ainvdiag), u), axis=1)), tf.reduce_prod(Adiag, axis=1));
        g_log_det_jacobian = tf.log(g_det_jacobian);
        sum_log_det_jacobians += g_log_det_jacobian;
        Z = tf.concat((Z, tf.expand_dims(1-tf.reduce_sum(Z, axis=1), 1)), axis=1);

    return Z, sum_log_det_jacobians;


def adam_updates(params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999):
    ''' Adam optimizer '''
    updates = [];
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params);
    else:
        grads = cost_or_grads
    t = tf.Variable(1., 'adam_t')
    for p, g in zip(params, grads):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
        if mom1>0:
            v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')
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
        gamma_t = tf.zeros((D,D));
        for k in range(1, min(t,K)+1):
            gamma_t += tf.matmul(As[k-1], gamma[t-k]);
        gamma.append(gamma_t);

    # compute the off-block-diagonal covariance blocks
    # first row
    for s in range(1, T):
        zcov_0s = tf.zeros((D,D));
        for k in range(1, min(s, K)+1):
            tau = s-k;
            zcov_0tau = zcov[0][tau];
            zcov_0s += tf.matmul(zcov_0tau, tf.transpose(As[k-1]));
        zcov[0].append(zcov_0s);
        zcov.append([tf.transpose(zcov_0s)]);

    # remaining rows
    for t in range(1,T):
        for s in range(t, T):
            zcov_ts = tf.zeros((D,D));
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
        diffmat = np.zeros((T,T), dtype=np.float32);
        for i in range(T):
            for j in range(T):
                diffmat_ij = float(i-j);
                diffmat[i,j] = diffmat_ij;
                if (i is not j):
                    diffmat[j,i] = diffmat_ij;
        GPcovd = np.exp(-np.square(diffmat) / (2.0*np.square(l[0]))) + eps*np.eye(T, dtype=np.float32);
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

def factors(n):
    return [f for f in range(1,n+1) if n%f==0]
