import tensorflow as tf
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
import datetime
import scipy.stats
import sys
import os
import io
from sklearn.metrics import pairwise_distances
from statsmodels.tsa.ar_model import AR
from efn_util import MMD2u, PlanarFlowLayer, computeMoments, getEtas, \
                      latent_dynamics, time_invariant_flow, construct_flow, \
                      initialize_optimization_parameters, load_constraint_info, \
                      approxKL

def train_network(constraint_id, flow_id, cost_type,  L=1, n_k=50, lr_order=-3, random_seed=0, param_network=False):
    D_Z, K_eta, params, constraint_type = load_constraint_info(constraint_id);
    layers, num_linear_lyrs, num_planar_lyrs, K = construct_flow(flow_id, D_Z);
    assert(isinstance(K, int) and K >= 0);
    dynamics = K > 0;
    T = 1; # let's generalize to processes later :P (not within scope of NIPS submission)
    n = n_k*K_eta; 

    # optimization hyperparameters
    num_iters = 100000;
    opt_method = 'adam';
    lr = 10**lr_order
    check_diagnostics_rate = 1;

    np.random.seed(0);

    # good practice
    tf.reset_default_graph();
    tf.set_random_seed(random_seed);

    # save tensorboard summary in intervals
    tb_save_every = 10;

    #sigma_eps_buf = 1e-8;
    tb_save_flow_param = False;
    np_save_flow_param = True;


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


    if (constraint_type == 'dirichlet'):
        D_X = D_Z+1;
    else:
        D_X = D_Z;

    savedir = resdir + '/tb/' + '%s_%s_%s_seed=%d/' % (constraint_id, flow_id, cost_type, random_seed);

    if (constraint_type == 'normal'):
        mu_targs = params['mu_targs'];
        Sigma_targs = params['Sigma_targs'];
        mu_OL_targs = params['mu_OL_targs'];
        Sigma_OL_targs = params['Sigma_OL_targs'];
    elif (constraint_type == 'dirichlet'):
        alpha_targs = params['alpha_targs'];
        alpha_OL_targs = params['alpha_OL_targs'];

    # get etas based on constraint_id
    _etas, _off_lattice_etas = getEtas(constraint_id, K_eta);
    ncons = _etas[0].shape[0];

    _eta = np.zeros((n, ncons));
    _off_lattice_eta = np.zeros((n,ncons));
    for k in range(K_eta):
        for i in range(n_k):
            k_start = k*n_k;
            _eta[k_start+i,:] = _etas[k][:,0];
            _off_lattice_eta[k_start+i,:] = _off_lattice_etas[k][:,0];

    # Placeholder for initial condition of latent dynamical process
    if (dynamics):
        num_zi = 1;
    else:
        num_zi = T;
    Z0 = tf.placeholder(tf.float32, shape=(None, D_Z, num_zi));
    eta = tf.placeholder(tf.float32, shape=(None, ncons));
    batch_size = tf.shape(Z0)[0];

    # construct the parameter network
    nlayers = len(layers);
    if (param_network):
        h = eta;
        for i in range(L):
            with tf.variable_scope('ParamNetLayer%d' % (i+1)):
                h = tf.layers.dense(h, ncons); # each layer will have ncons nodes (can change this)
        theta = [];
        for i in range(nlayers):
            layer = layers[i];
            layer_name, param_names, param_dims = layer.get_layer_info();
            #print(layer_name, param_names, param_dims);
            nparams = len(param_names);
            layer_i_params = [];
            # read each parameter out of the last layer.
            for j in range(nparams):
                num_elems = np.prod(param_dims[j]);
                A_shape = (ncons, num_elems);
                b_shape = (1, num_elems);
                A_ij = tf.get_variable(layer_name+'_'+param_names[j]+'_A', shape=A_shape, \
                                           dtype=tf.float32, \
                                           initializer=tf.glorot_uniform_initializer());
                b_ij = tf.get_variable(layer_name+'_'+param_names[j]+'_b', shape=b_shape, \
                                           dtype=tf.float32, \
                                           initializer=tf.glorot_uniform_initializer());
                param_ij = tf.matmul(h, A_ij) + b_ij;
                param_ij = tf.reshape(param_ij, (n,) + param_dims[j]);
                layer_i_params.append(param_ij);
            theta.append(layer_i_params);


    else: # or just declare the parameters
        theta =[];
        for i in range(nlayers):
            layer = layers[i];
            layer_name, param_names, param_dims = layer.get_layer_info();
            #print(layer_name, param_names, param_dims);
            nparams = len(param_names);
            layer_i_params = [];
            for j in range(nparams):
                param_ij = tf.get_variable(layer_name+'_'+param_names[j], shape=param_dims[j], \
                                           dtype=tf.float32, \
                                           initializer=tf.glorot_uniform_initializer());
                layer_i_params.append(param_ij);
            theta.append(layer_i_params);

    if (dynamics):
        # Note: The vector autoregressive parameters are vectorized for convenience in the 
        # unbiased gradient computation. 
        # Update, no need to vectorize. Fix this when implementing dynamics
        A_init = tf.random_normal((K*D_Z*D_Z,1), 0.0, 1.0, dtype=tf.float32);
        #A_init = np.array([[0.5], [0.0], [0.0], [0.5]], np.float32);
        log_sigma_eps_init = tf.abs(tf.random_normal((D,1), 0.0, 0.1, dtype=tf.float32));
        noise_level = np.log(.02);
        log_sigma_eps_init = noise_level*np.array([[1.0], [1.0]], np.float32);
        A_vectorized = tf.get_variable('A_vectorized', dtype=tf.float32, initializer=A_init);
        A = tf.reshape(A_vectorized, [K,D_Z,D_Z]);
        log_sigma_eps = tf.get_variable('log_sigma_eps', dtype=tf.float32, initializer=log_sigma_eps_init);
        sigma_eps = tf.exp(log_sigma_eps);
        num_dyn_param_vals = K*D_Z*D_Z + D_Z;
        # contruct latent dynamics 
        Z_AR, base_log_p_z, Sigma_Z_AR = latent_dynamics(Z0, A, sigma_eps, T);
    else:
        # evaluate unit normal 
        p0 = tf.reduce_prod(tf.exp((-tf.square(Z0))/2.0)/tf.sqrt(2.0*np.pi), axis=1); 
        base_log_p_z = tf.log(p0);
        # TODO more care taken care for time series
        num_dyn_param_vals = 0;
        Z_AR = Z0;

    # construct time-invariant 
    if (nlayers > 0):
        Z, sum_log_det_jacobian = time_invariant_flow(Z_AR, layers, theta, constraint_type);
    else:
        Z = Z_AR;
        sum_log_det_jacobian = tf.zeros((batch_size*T,1));

    # generative model is fully specified
    all_params = tf.trainable_variables();
    nparams = len(all_params);

    log_p_zs = base_log_p_z - sum_log_det_jacobian;

    X = tf.transpose(tf.reshape(Z, [batch_size, T, D_X]), [0, 2, 1]); # should be [n,D,T] now
 
    X_cov = tf.div(tf.matmul(X, tf.transpose(X, [0, 2, 1])), T); # this is [n x D x D]

    # set up the constraint computation
    Tx = computeMoments(X, constraint_id);

    # exponential family optimization
    y = log_p_zs;
    y_mc = y - tf.reduce_mean(y);
    var_y = tf.reduce_mean(tf.square(y_mc));
    TSS = tf.reduce_sum(tf.square(y_mc));
    cost = 0.0;
    R2s = [];
    for k in range(K_eta):
        # get eta-specific log-probs and T(x)'s
        k_start = k*n_k;
        k_end = (k+1)*n_k;
        y_k = y[k_start:k_end,:];
        Tx_k = Tx[k_start:k_end,:];
        _eta_k = _etas[k];
        # compute optimial linear regression offset term for eta
        alpha_k = tf.reduce_mean(y_k - tf.matmul(Tx_k, _eta_k));
        residuals = y_k - tf.matmul(Tx_k, _eta_k) - alpha_k;
        RSS_k = tf.matmul(tf.transpose(residuals), residuals);
        y_k_mc = y_k - tf.reduce_mean(y_k);
        TSS_k = tf.reduce_sum(tf.square(y_k_mc));

        # compute the R^2 of the exponential family fit
        R2s.append(1.0 - (RSS_k[0,0] / TSS_k));

        if (cost_type == 'reg'):
            cost += RSS_k[0,0];
        elif (cost_type == 'KL'):
            cost += tf.reduce_mean(y_k - tf.matmul(Tx_k, _eta_k));

    cost_grad = tf.gradients(cost, all_params);

    # set optimization hyperparameters
    saver = tf.train.Saver();
    tf.add_to_collection('Z0', Z0);
    tf.add_to_collection('X', X);

    # tensorboard logging
    summary_writer = tf.summary.FileWriter(savedir);
    for k in range(K_eta):
        tf.summary.scalar('R2%d' % (k+1), R2s[k]);
    tf.summary.scalar('cost', cost);
    tf.summary.histogram('log_p(z)', y[:,0]);
    tf.summary.scalar('var log p(z)', TSS);

    # log parameter values throughout optimization

    num_flow_param_vals = 0;
    flow_params = [];
    flow_grads = []
    flow_param_names = [];
    flow_param_nvars = [];
    for i in range(nparams):
        param = all_params[i];
        param_shape = tuple(param.get_shape().as_list());
        if ((not dynamics) or i > 1):
            flow_param_names.append(param.name);
            # all flow network parameters are vectorized
            flow_param_nvars.append(param.shape[0]);
            num_flow_param_vals += param_shape[0];
            flow_params.append(param);
        if (tb_save_flow_param):
            assert(param_shape[1] == 1);
            for ii in range(param_shape[0]):
                for jj in range(param_shape[1]):
                    if (param_shape[1]==1):
                        tf.summary.scalar('%s_%d' % (param.name[:-2], ii+1), param[ii, jj]);
                    else:
                        tf.summary.scalar('%s_%d%d' % (param.name[:-2], ii+1, jj+1), param[ii, jj]);
    num_flow_params = len(flow_params);

    summary_op = tf.summary.merge_all()

    array_init_len = num_iters;
    if (dynamics):
        As = np.zeros((array_init_len, K, D_Z, D_Z));
        sigma_epsilons = np.zeros((array_init_len,D_Z));
    X_covs = np.zeros((array_init_len, D_X, D_X));
    flow_param_vals = np.zeros((array_init_len, num_flow_param_vals,));
    cost_grad_vals = np.zeros((array_init_len, num_dyn_param_vals + num_flow_param_vals));
    array_cur_len = array_init_len;

    num_diagnostic_checks = (num_iters // check_diagnostics_rate) + 1;
    train_R2s = np.zeros((num_diagnostic_checks, K_eta));
    test_R2s = np.zeros((num_diagnostic_checks, K_eta));
    train_KLs = np.zeros((num_diagnostic_checks, K_eta));
    test_KLs = np.zeros((num_diagnostic_checks, K_eta));
    check_it = 0;
    with tf.Session() as sess:
        if (param_network):
            print('D=%d, T=%d, K=%d, n=%d, lr=10^%.1f' % (D_Z, T, K, n, np.log10(lr)));
        grads_and_vars = [];
        for i in range(len(all_params)):
            grads_and_vars.append((cost_grad[i], all_params[i]));
        init_op = tf.global_variables_initializer();
        sess.run(init_op);

        z_i = np.random.normal(np.zeros((n,D_Z,num_zi)), 1.0);
        feed_dict = {Z0:z_i, eta:_eta};
        if (dynamics):
            cost_i, A_i, X_cov_i, _sigma_epsilon_i, _flow_params, _cost_grads, summary = \
                    sess.run([cost, A, X_cov, sigma_eps, flow_params, cost_grad, summary_op], feed_dict);
            As[0,:,:,:] = A_i;
            sigma_epsilons[0,:] = _sigma_epsilon_i[:,0];
            X_covs[0, :, :] = np.mean(X_cov_i, 0);
        else:
            cost_i, X_cov_i, _flow_params, _cost_grads, _X, _y, _Tx, summary = \
                    sess.run([cost, X_cov, flow_params, cost_grad, X, y, Tx, summary_op], feed_dict);
            X_covs[0, :, :] = np.mean(X_cov_i, 0);

        #np.savez('notebooks/debug.npz', Z0=z_i, X=_X, y=_y, Tx=_Tx, eta=_etas[0]);
        fpv_ind = 0;
        for j in range(num_flow_params):
            param = _flow_params[j];
            param_len = len(param);
            for jj in range(param_len):
                if (len(param.shape) == 1):
                    flow_param_vals[0,fpv_ind] = param[jj];
                else:
                    flow_param_vals[0,fpv_ind] = param[jj,0];
                fpv_ind += 1;
        assert(fpv_ind == num_flow_param_vals);

        cgv_ind = 0;
        for j in range(nparams):
            grad = _cost_grads[j];
            grad_len = len(grad);
            for jj in range(grad_len):
                if (len(grad.shape)==1):
                    cost_grad_vals[0, cgv_ind] = grad[jj];
                else:
                    cost_grad_vals[0, cgv_ind] = grad[jj,0];
                cgv_ind += 1;

        # reset optimizer
        if (opt_method == 'adam'):
            optimizer = tf.train.AdamOptimizer(learning_rate=lr);
        elif (opt_method == 'adadelta'):
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr);
        elif (opt_method == 'adagrad'):
            optimizer = tf.train.AdagradOptimizer(learning_rate=lr);
        elif (opt_method == 'graddesc'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr);

        train_step = optimizer.apply_gradients(grads_and_vars);
        initialize_optimization_parameters(sess, optimizer, all_params);
        slot_names = optimizer.get_slot_names();
        debug_opt_var = optimizer.get_slot(all_params[0], slot_names[0]);
        # SGD iteration
        i = 1;

        #log_param_diff = log_param_diff_thresh + 1;
        has_converged = False;
        #while ((i < (cost_grad_lag)) or (not has_converged)): 
        convergence_it = 0;
        while (i < num_iters):
            if (i == array_cur_len):
                if (dynamics):
                    As = np.concatenate((As, np.zeros((array_cur_len, K, D_Z, D_Z))), axis=0);
                    sigma_epsilons = np.concatenate((sigma_epsilons, np.zeros((array_cur_len,D))), axis=0);
                X_covs = np.concatenate((X_covs, np.zeros((array_cur_len, D_X, D_X))), axis=0);
                if (np_save_flow_param):
                    flow_param_vals = np.concatenate((flow_param_vals, np.zeros((array_cur_len, num_flow_param_vals))), axis=0);
                    cost_grad_vals = np.concatenate((cost_grad_vals, np.zeros((array_cur_len, num_flow_param_vals))), axis=0);
                
                # double array lengths
                array_cur_len = 2*array_cur_len;

            z_i = np.random.normal(np.zeros((n,D_Z,num_zi)), 1.0);
            feed_dict = {Z0:z_i, eta:_eta};
            if (dynamics):
                ts, cost_i, A_i, X_cov_i, _sigma_epsilon_i, _X, _flow_params, _cost_grads, _R2s, summary = \
                    sess.run([train_step, cost, A, X_cov, sigma_eps, X, flow_params, cost_grad, R2s, summary_op], feed_dict);
                As[i,:,:] = A_i;
                sigma_epsilons[i,:] = _sigma_epsilon_i[:,0];
                X_covs[i, :, :] = np.mean(X_cov_i, 0);
            else:
                ts, cost_i, X_cov_i, _X, _flow_params, _cost_grads, _R2s, _log_p_zs, _Tx, summary = \
                    sess.run([train_step, cost, X_cov, X, flow_params, cost_grad, R2s, log_p_zs, Tx, summary_op], feed_dict);
                X_covs[i, :, :] = np.mean(X_cov_i, 0);

            if (np_save_flow_param):
                fpv_ind = 0;
                for j in range(num_flow_params):
                    param = _flow_params[j];
                    param_len = len(param);
                    for jj in range(param_len):
                        if (len(param.shape)==1):
                            flow_param_vals[i,fpv_ind] = param[jj];
                        else:
                            flow_param_vals[i,fpv_ind] = param[jj,0];
                        fpv_ind += 1;

                cgv_ind = 0;
                for j in range(nparams):
                    grad = _cost_grads[j];
                    grad_len = len(grad);
                    for jj in range(grad_len):
                        if (len(grad.shape)==1):
                            cost_grad_vals[i, cgv_ind] = grad[jj];
                        else:
                            cost_grad_vals[i, cgv_ind] = grad[jj,0];
                        cgv_ind += 1;

            if (np.mod(i,tb_save_every)==0):
                summary_writer.add_summary(summary, i);

            if (np.mod(i, check_diagnostics_rate)==0):
                """
                plt.figure();
                plt.scatter(_X[:,0,0], _X[:,1,0], 40*np.ones((n,)), _Tx[:,4]);
                plt.colorbar();
                plt.show();
                """

                train_R2s[check_it,:] = _R2s;
                # compute KL
                training_KL = [];
                for k in range(K_eta):
                    k_start = k*n_k;
                    k_end = (k+1)*n_k;
                    _y_k = _log_p_zs[k_start:k_end];
                    _X_k = _X[k_start:k_end, :, :];
                    _X_k = np.reshape(np.transpose(_X_k, [0, 2, 1]), [T*n_k, D_X]);
                    if (constraint_type == 'normal'):
                        params_k = {'mu':mu_targs[k], 'Sigma':Sigma_targs[k]};
                    elif (constraint_type == 'dirichlet'):
                        params_k = {'alpha':alpha_targs[k]};
                    training_KL.append(approxKL(_y_k, _X_k, constraint_type, params_k));
                train_KLs[check_it,:] = training_KL;

                feed_dict = {Z0:z_i, eta:_off_lattice_eta};
                _X_off_lattice, _log_p_zs, _test_R2s  = sess.run([X, log_p_zs, R2s], feed_dict);
                test_R2s[check_it,:] = _test_R2s;

                # compute KL
                testing_KL = [];
                for k in range(K_eta):
                    k_start = k*n_k;
                    k_end = (k+1)*n_k;
                    _y_k = _log_p_zs[k_start:k_end];
                    _X_k = _X_off_lattice[k_start:k_end, :, :];
                    _X_k = np.reshape(np.transpose(_X_k, [0, 2, 1]), [T*n_k, D_X]);
                    if (constraint_type == 'normal'):
                        params_k = {'mu':mu_OL_targs[k], 'Sigma':Sigma_OL_targs[k]};
                    elif (constraint_type == 'dirichlet'):
                        params_k = {'alpha':alpha_OL_targs[k]};
                    testing_KL.append(approxKL(_y_k, _X_k, constraint_type, params_k));
                test_KLs[check_it,:] = testing_KL;

                if (param_network):
                    print(42*'*');
                    print('it = %d ' % i);
                    print('cost', cost_i);
                    print('training R2', _R2s);
                    print('training KL', training_KL);
                    print('test R2', _test_R2s);
                    print('testing KL', testing_KL);
                    print('saving to %s  ...' % savedir);
                    print(42*'*');
                if (np_save_flow_param):
                    if (dynamics):
                        np.savez(savedir + 'results.npz', As=As, sigma_epsilons=sigma_epsilons, flow_param_vals=flow_param_vals, cost_grad_vals=cost_grad_vals, \
                                                          flow_param_names=flow_param_names,  flow_param_nvars=flow_param_nvars, \
                                                          X_covs=X_covs, autocov_targ=autocov_targ, it=i, \
                                                          X=_X, X_off_lattice=_X_off_lattice, \
                                                          train_R2s=train_R2s, test_R2s=test_R2s, train_KLs=train_KLs, test_KLs=test_KLs);
                    else:
                        np.savez(savedir + 'results.npz', flow_param_vals=flow_param_vals, flow_param_names=flow_param_names,  cost_grad_vals=cost_grad_vals, \
                                                          flow_param_nvars=flow_param_nvars, X_covs=X_covs, it=i, \
                                                          X=_X, Tx=_Tx, log_p_zs=_log_p_zs, X_off_lattice=_X_off_lattice, \
                                                          train_R2s=train_R2s, test_R2s=test_R2s, train_KLs=train_KLs, test_KLs=test_KLs);
                else:
                    if (dynamics):
                        np.savez(savedir + 'results.npz', As=As, sigma_epsilons=sigma_epsilons, X_covs=X_covs, autocov_targ=autocov_targ,  \
                                                          it=i, X=_X, X_off_lattice=_X_off_lattice, \
                                                          train_R2s=train_R2s, test_R2s=test_R2s, train_KLs=train_KLs, test_KLs=test_KLs);
                    else:
                        np.savez(savedir + 'results.npz', X_covs=X_covs, it=i, \
                                                          X=_X, X_off_lattice=_X_off_lattice, \
                                                          train_R2s=train_R2s, test_R2s=test_R2s, train_KLs=train_KLs, test_KLs=test_KLs);
                
                check_it += 1;

                if (not param_network):
                    if (_R2s[0] > .9999):
                        print('Test successful!');
                        print('We can learn the %s distribution with a %s flow network' % (constraint_id, flow_id));
                        break;
           

            sys.stdout.flush();


            i += 1;

        # save all the hyperparams
        if not os.path.exists(savedir):
                print('Making directory %s' % savedir);
                os.makedirs(savedir);
        #saveParams(params, savedir);
        # save the model
        saver.save(sess, savedir + 'model');
    return _X, _R2s;

if __name__ == '__main__':    # parse command line parameters
    n_args = len(sys.argv);
    constraint_id = str(sys.argv[1]);
    flow_id = str(sys.argv[2]);
    cost_type = str(sys.argv[3]);
    L = int(sys.argv[4]);
    if (n_args > 5):
        n_k= int(sys.argv[5]);
    else:
        n_k = 50;
    if (n_args > 6):
        lr_order = float(sys.argv[6]);
    else:
        lr_order = -3;
    if (n_args > 7):
        random_seed = int(sys.argv[7]);
    else:
        random_seed = 0;
    if (n_args > 8):
        param_network_input = sys.argv[8];
        param_network = not (str(param_network_input) == 'False');
    else:
        param_network = False;

    train_network(constraint_id, flow_id, cost_type, L, n_k, lr_order, random_seed, param_network);
