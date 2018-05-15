import tensorflow as tf
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
import scipy.stats
import sys
import os
import io
from sklearn.metrics import pairwise_distances
from statsmodels.tsa.ar_model import AR
from efn_util import MMD2u, PlanarFlowLayer, computeMoments, \
                      latent_dynamics, connect_flow, construct_flow, \
                      setup_IO, normal_eta, log_grads, inv_wishart_eta, prp_tn_eta, dir_dir_eta, \
                      approxKL, drawEtas, checkH, declare_theta, cost_fn, \
                      computeLogBaseMeasure, check_convergence, batch_diagnostics, \
                      memory_extension, setup_param_logging, count_params, get_ef_dimensionalities

def train_nf(exp_fam, params, flow_dict, cost_type, M_eta=100, \
               lr_order=-3, random_seed=0, max_iters=10000, check_rate=1000):

    T = 1; # let's generalize to processes later :P (not within scope of NIPS submission)
    stop_early = False;
    cost_grad_lag = check_rate;
    pthresh = 0.1;
 
    D = params['D'];

    D_Z, ncons, _, num_Tx_inputs = get_ef_dimensionalities(exp_fam, D, False);

    # good practice
    tf.reset_default_graph();

    flow_layers, Z0, Z_AR, base_log_p_z, P, num_zi, num_theta_params, num_dyn_param_vals = construct_flow(exp_fam, flow_dict, D_Z, T);
    K = tf.shape(Z0)[0];
    M = tf.shape(Z0)[1];
    batch_size = tf.multiply(K, M);
    dynamics = P > 0;

    K_eta = 1;
    n = K_eta*M_eta;

    # optimization hyperparameters
    lr = 10**lr_order
    # save tensorboard summary in intervals
    model_save_every = 49999;
    tb_save_every = 50;
    tb_save_params = False;

    # seed RNGs
    np.random.seed(0);
    tf.set_random_seed(random_seed);

    savedir = setup_IO(exp_fam, K_eta, M_eta, D, flow_dict, {}, False, False, random_seed);
    eta = tf.placeholder(tf.float64, shape=(None, ncons));
    Tx_input = tf.placeholder(tf.float64, shape=(None, num_Tx_inputs));

    if (exp_fam == 'normal'):
        mu_targ =  params['mu'];
        Sigma_targ = params['Sigma'];
        _eta, _ = normal_eta(mu_targ[0], Sigma_targ[0], False);
        _Tx_input = np.zeros((K_eta,num_Tx_inputs));
    elif (exp_fam == 'dirichlet'):
        alpha_targ = params['alpha'];
        _eta = alpha_targ;
        _Tx_input = np.zeros((K_eta,num_Tx_inputs));
    elif (exp_fam == 'inv_wishart'):
        Psi_targ = params['Psi'];
        m_targ = params['m'];
        _eta, _ = inv_wishart_eta(Psi_targ[0], m_targ[0,0], False);
        _Tx_input = np.zeros((K_eta,num_Tx_inputs));
    elif (exp_fam == 'prp_tn'):
        mus =  params['mus'];
        Sigmas = params['Sigmas'];
        xs = params['xs'];
        Ns = params['Ns'];
        _eta, _ = prp_tn_eta(mus[0], Sigmas[0], xs[0], Ns[0], False);
        _Tx_input = np.zeros((K_eta,num_Tx_inputs));
    elif (exp_fam == 'dir_dir'):
        alpha_0s = params['alpha_0s'];
        betas = params['betas'];
        xs = params['xs'];
        zs = params['zs'];
        Ns = params['Ns'];
        _eta, _ = dir_dir_eta(alpha_0s[0], xs[0], Ns[0], False);
        _Tx_input = np.array([[betas[0]]]);

    _eta_test = _eta;
    _Tx_input_test = _Tx_input;

    # construct the parameter network
    L_flow = len(flow_layers);
    theta = declare_theta(flow_layers);

    # connect time-invariant flow
    Z, sum_log_det_jacobian, Z_by_layer = connect_flow(Z_AR, flow_layers, theta);
    log_p_zs = base_log_p_z - sum_log_det_jacobian;

    # generative model is fully specified
    all_params = tf.trainable_variables();
    nparams = len(all_params);

    X = Z; # [n,D,T] 
    # set up the constraint computation
    Tx = computeMoments(X, exp_fam, D, T, Z_by_layer, Tx_input);
    Bx = computeLogBaseMeasure(X, exp_fam, D, T);
    # exponential family optimization
    cost, costs, R2s = cost_fn(eta, log_p_zs, Tx, Bx, K_eta, cost_type)
    cost_grad = tf.gradients(cost, all_params);

    grads_and_vars = [];
    for i in range(len(all_params)):
        grads_and_vars.append((cost_grad[i], all_params[i]));

    # set optimization hyperparameters
    tf.add_to_collection('Z0', Z0);
    tf.add_to_collection('X', X);
    tf.add_to_collection('eta', eta);
    tf.add_to_collection('log_p_zs', log_p_zs);
    tf.add_to_collection('Tx_input', Tx_input);
    saver = tf.train.Saver();

    # tensorboard logging
    summary_writer = tf.summary.FileWriter(savedir);
    for k in range(K_eta):
        tf.summary.scalar('R2%d' % (k+1), R2s[k]);
    tf.summary.scalar('cost', cost);

    # log parameter values throughout optimization
    if (tb_save_params):
        setup_param_logging(all_params);

    if (stop_early):
        nparam_vals = count_params(all_params);

    summary_op = tf.summary.merge_all()

    opt_compress_fac = 16;
    array_init_len = int(np.ceil(max_iters/opt_compress_fac));
    if (dynamics):
        As = np.zeros((array_init_len, K, D_Z, D_Z));
        sigma_epsilons = np.zeros((array_init_len,D_Z));
    if (stop_early):
        cost_grad_vals = np.zeros((array_init_len, nparam_vals));
    array_cur_len = array_init_len;

    num_diagnostic_checks = (max_iters // check_rate);
    train_elbos = np.zeros((num_diagnostic_checks, K_eta));
    test_elbos = np.zeros((num_diagnostic_checks, K_eta));
    train_R2s = np.zeros((num_diagnostic_checks, K_eta));
    test_R2s = np.zeros((num_diagnostic_checks, K_eta));
    train_KLs = np.zeros((num_diagnostic_checks, K_eta));
    test_KLs = np.zeros((num_diagnostic_checks, K_eta));
    train_elbos = np.zeros((num_diagnostic_checks, K_eta));
    check_it = 0;
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer();
        sess.run(init_op);

        z_i = np.random.normal(np.zeros((K_eta, M_eta, D_Z, num_zi)), 1.0);
        feed_dict = {Z0:z_i, eta:_eta, Tx_input:_Tx_input};

        cost_i, _cost_grads, _X, _y, _base_log_p_z, _Tx, summary = \
            sess.run([cost, cost_grad, X, log_p_zs, base_log_p_z, Tx, summary_op], feed_dict);

        if (dynamics):
            A_i, _sigma_epsilon_i = sess.run([A, sigma_eps]);
            As[0,:,:,:] = A_i;
            sigma_epsilons[0,:] = _sigma_epsilon_i[:,0];

        if (stop_early):
            log_grads(_cost_grads, cost_grad_vals, 0);

        optimizer = tf.train.AdamOptimizer(learning_rate=lr);

        train_step = optimizer.apply_gradients(grads_and_vars);
        
        init_op = tf.global_variables_initializer();
        sess.run(init_op);

        i = 1;

        has_converged = False;
        while (i < max_iters):
            if (stop_early):
                has_converged = check_convergence([cost_grad_vals], i, cost_grad_lag, pthresh, criteria='grad_mean_ttest');
                
            if (stop_early and i == array_cur_len):
                cost_grad_vals = memory_extension(cost_grad_vals, array_cur_len);
                array_cur_len = 2*array_cur_len;

            z_i = np.random.normal(np.zeros((K_eta, M_eta, D_Z, num_zi)), 1.0);
            feed_dict = {Z0:z_i, eta:_eta, Tx_input:_Tx_input};

            if (np.mod(i, check_rate)==0):
                start_time = time.time();
            ts, cost_i, _X, _cost_grads, _R2s, _Tx, _Bx, summary = \
                    sess.run([train_step, cost, X, cost_grad, R2s, Tx, Bx, summary_op], feed_dict);

            if (np.mod(i, check_rate)==0):
                end_time = time.time();
                print('iter %d took %f seconds' % (i+1, end_time-start_time));

            if (dynamics):
                A_i, _sigma_epsilon_i = sess.run([A, sigma_eps]);
                As[i,:,:] = A_i;
                sigma_epsilons[i,:] = _sigma_epsilon_i[:,0];

            if (stop_early):
                log_grads(_cost_grads, cost_grad_vals, i);

            if (np.mod(i,tb_save_every)==0):
                summary_writer.add_summary(summary, i);

            if (np.mod(i,model_save_every) == 0):
                # save all the hyperparams
                if not os.path.exists(savedir):
                    print('Making directory %s' % savedir );
                    os.makedirs(savedir);
                print('saving model at iter', i);
                saver.save(sess, savedir + 'model');

            if (np.mod(i+1, check_rate)==0 and i > 0):
                if (stop_early):
                    has_converged = check_convergence([cost_grad_vals], i, cost_grad_lag, pthresh, criteria='grad_mean_ttest');
                
                z_i = np.random.normal(np.zeros((K_eta, int(1e3), D_Z, num_zi)), 1.0);
                feed_dict_train = {Z0:z_i, eta:_eta, Tx_input:_Tx_input};
                feed_dict_test = {Z0:z_i, eta:_eta_test, Tx_input:_Tx_input_test};
                train_costs_i, train_R2s_i, train_KLs_i = batch_diagnostics(exp_fam, K_eta, sess, feed_dict_train, X, log_p_zs, costs, R2s, params);
                test_costs_i, test_R2s_i, test_KLs_i = batch_diagnostics(exp_fam, K_eta, sess, feed_dict_test, X, log_p_zs, costs, R2s, params);

                train_elbos[check_it,:] = np.array(train_costs_i);
                test_elbos[check_it,:] = np.array(test_costs_i);
                train_R2s[check_it,:] = np.array(train_R2s_i);
                train_KLs[check_it,:] = np.array(train_KLs_i);
                test_R2s[check_it,:] = np.array(test_R2s_i);
                test_KLs[check_it,:] = np.array(test_KLs_i);

                mean_train_elbo = np.mean(train_costs_i);
                mean_train_R2 = np.mean(train_R2s_i);
                mean_train_KL = np.mean(train_KLs_i);

                print(42*'*');
                print('it = %d ' % (i+1));
                print('cost = %f ' % cost_i);
                print('train elbo %.3f, train R2: %.3f, train KL %.3f' % (mean_train_elbo, mean_train_R2, mean_train_KL));
                if (dynamics):
                    np.savez(savedir + 'results.npz', As=As, sigma_epsilons=sigma_epsilons, autocov_targ=autocov_targ,  \
                                                      it=i, X=_X, eta=_eta, Tx_input=_Tx_input, params=params, check_rate=check_rate, \
                                                      train_elbos=train_elbos, test_elbos=test_elbos, \
                                                      train_R2s=train_R2s, test_R2s=test_R2s, \
                                                      train_KLs=train_KLs, test_KLs=test_KLs);
                else:
                    np.savez(savedir + 'results.npz', it=i, X=_X, eta=_eta, Tx_input=_Tx_input, params=params, check_rate=check_rate, \
                                                      train_elbos=train_elbos, test_elbos=test_elbos, \
                                                      train_R2s=train_R2s, test_R2s=test_R2s, \
                                                      train_KLs=train_KLs, test_KLs=test_KLs);
                
                check_it += 1;

            sys.stdout.flush();
            i += 1;

        z_i = np.random.normal(np.zeros((K_eta, M_eta, D_Z, num_zi)), 1.0);
        feed_dict = {Z0:z_i, eta:_eta, Tx_input:_Tx_input};
        _log_p_zs, _X = sess.run([log_p_zs, X], feed_dict);

    if (len(_X.shape) > 2):
        assert(len(_X.shape) == 4);
        _X = _X[0, :, :, 0];
    return _log_p_zs, _X, train_R2s, train_KLs, i;
