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
                      setup_IO, construct_param_network, log_grads, \
                      approxKL, drawEtas, checkH, declare_theta, cost_fn, \
                      computeLogBaseMeasure, check_convergence, batch_diagnostics, \
                      memory_extension, setup_param_logging, count_params, \
                      get_param_network_hyperparams, get_ef_dimensionalities
from tensorflow.python import debug as tf_debug

def train_efn(exp_fam, D, flow_dict, cost_type, K_eta, M_eta, stochastic_eta, \
              give_inverse_hint=False, lr_order=-3, random_seed=0, \
              max_iters=10000, check_rate=200):
    batch_norm = False;
    dropout = False;
    upl_tau = None;
    upl_shape = 'linear';
    T = 1; # let's generalize to processes later (not within scope of NIPS submission)
    stop_early = False;
    cost_grad_lag = 100;
    pthresh = 0.1;

    # seed RNGs
    np.random.seed(random_seed);
    tf.set_random_seed(random_seed);

    D_Z, ncons, num_param_net_inputs = get_ef_dimensionalities(exp_fam, D, give_inverse_hint);

    # set number of layers in the parameter network
    L = 8;
    #L = max(int(np.ceil(np.sqrt(D_Z))), 4);  # we use at least four layers

    # good practice
    tf.reset_default_graph();

    flow_layers, Z0, Z_AR, base_log_p_z, P, num_zi, num_theta_params, num_dyn_param_vals = construct_flow(exp_fam, flow_dict, D_Z, T);
    K = tf.shape(Z0)[0];
    M = tf.shape(Z0)[1];
    batch_size = tf.multiply(K, M);
    dynamics = P > 0;

    n = K_eta*M_eta;

    # optimization hyperparameters
    opt_method = 'adam';
    lr = 10**lr_order
    # save tensorboard summary in intervals
    tb_save_every = 50;
    model_save_every = 1000;
    tb_save_params = False;

    eta = tf.placeholder(tf.float64, shape=(None, ncons));
    param_net_input = tf.placeholder(tf.float64, shape=(None, num_param_net_inputs));

    if (not stochastic_eta):
        # get etas based on constraint_id
        _eta, _param_net_input, eta_draw_params = drawEtas(exp_fam, D, K_eta, give_inverse_hint);
        _eta_test, _param_net_input_test, eta_test_draw_params = drawEtas(exp_fam, D, K_eta, give_inverse_hint);


    param_net_hps = get_param_network_hyperparams(L, num_param_net_inputs, num_theta_params, upl_tau, upl_shape);

    savedir = setup_IO(exp_fam, K_eta, M_eta, D, flow_dict, param_net_hps, stochastic_eta, give_inverse_hint, random_seed);
    print(random_seed, savedir);

    # construct the parameter network
    theta = construct_param_network(param_net_input, K_eta, flow_layers, param_net_hps);

    # connect time-invariant flow
    Z, sum_log_det_jacobian, Z_by_layer = connect_flow(Z_AR, flow_layers, theta, exp_fam);
    log_p_zs = base_log_p_z - sum_log_det_jacobian;

    # generative model is fully specified
    all_params = tf.trainable_variables();
    nparams = len(all_params);

    X = Z; # [n,D,T] 
    # set up the constraint computation
    Tx = computeMoments(X, exp_fam, D, T, Z_by_layer);
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
    tf.add_to_collection('param_net_input', param_net_input);
    tf.add_to_collection('log_p_zs', log_p_zs);
    saver = tf.train.Saver();

    # tensorboard logging
    summary_writer = tf.summary.FileWriter(savedir);
    #for k in range(K_eta):
    #    tf.summary.scalar('R2%d' % (k+1), R2s[k]);
    tf.summary.scalar('cost', cost);

    if (tb_save_params):
        setup_param_logging(all_params);

    if (stop_early):
        nparam_vals = count_params(all_params);

    summary_op = tf.summary.merge_all();

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
    check_it = 0;
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer();
        sess.run(init_op);

        z_i = np.random.normal(np.zeros((K_eta, M_eta, D_Z, num_zi)), 1.0);
        if (stochastic_eta):
            _eta, _param_net_input, eta_draw_params = drawEtas(exp_fam, D, K_eta, give_inverse_hint);
            _eta_test, _param_net_input_test, eta_test_draw_params = drawEtas(exp_fam, D, K_eta, give_inverse_hint);
        feed_dict = {Z0:z_i, eta:_eta, param_net_input:_param_net_input};

        cost_i, _cost_grads, _X, _y, _Tx, summary = \
            sess.run([cost, cost_grad, X, log_p_zs, Tx, summary_op], feed_dict);

        if (dynamics):
            A_i, _sigma_epsilon_i = sess.run([A, sigma_eps]);
            As[0,:,:,:] = A_i;
            sigma_epsilons[0,:] = _sigma_epsilon_i[:,0];

        if (stop_early):
            log_grads(_cost_grads, cost_grad_vals, 0);

        optimizer = tf.train.AdamOptimizer(learning_rate=lr);

        train_step = optimizer.apply_gradients(grads_and_vars);
        #check_op = tf.add_check_numerics_ops();
        
        init_op = tf.global_variables_initializer();
        sess.run(init_op);

        i = 1;

        print('starting opt');
        has_converged = False;
        while (i < max_iters):
            if (stop_early and i == array_cur_len):
                memory_extension(cost_grad_vals, array_cur_len);
                array_cur_len = 2*array_cur_len;

            z_i = np.random.normal(np.zeros((K_eta, M_eta, D_Z, num_zi)), 1.0);
            if (stochastic_eta): 
                _eta, _param_net_input, eta_draw_params = drawEtas(exp_fam, D, K_eta, give_inverse_hint);

            feed_dict = {Z0:z_i, eta:_eta, param_net_input:_param_net_input};

            if (np.mod(i, check_rate)==0):
                start_time = time.time();
            ts, cost_i, _X, _cost_grads, _log_p_zs, _Tx, summary = \
                sess.run([train_step, cost, X, cost_grad, log_p_zs, Tx, summary_op], feed_dict);
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

            if (np.mod(i,model_save_every)==0):
                # save all the hyperparams
                if not os.path.exists(savedir):
                    print('Making directory %s' % savedir );
                    os.makedirs(savedir);
                #saveParams(params, savedir);
                # save the model
                saver.save(sess, savedir + 'model');

            if (np.mod(i+1, check_rate)==0):
                print(42*'*');
                print('it = %d ' % (i+1));
                start_time = time.time();
                if (stop_early):
                    has_converged = check_convergence([cost_grad_vals], i, cost_grad_lag, pthresh, criteria='grad_mean_ttest');
                
                # compute R^2 and KL for training and batch
                z_i = np.random.normal(np.zeros((K_eta, int(1e3), D_Z, num_zi)), 1.0);
                feed_dict_train = {Z0:z_i, eta:_eta, param_net_input:_param_net_input};
                feed_dict_test = {Z0:z_i, eta:_eta_test, param_net_input:_param_net_input_test};

                train_costs_i, train_R2s_i, train_KLs_i = batch_diagnostics(exp_fam, K_eta, sess, feed_dict_train, X, log_p_zs, costs, R2s, eta_draw_params);
                test_costs_i, test_R2s_i, test_KLs_i = batch_diagnostics(exp_fam, K_eta, sess, feed_dict_test, X, log_p_zs, costs, R2s, eta_test_draw_params);
                end_time = time.time();
                print('check diagnostics processes took: %f seconds' % (end_time-start_time));

                train_elbos[check_it,:] = np.array(train_costs_i);
                test_elbos[check_it,:] = np.array(test_costs_i);
                train_R2s[check_it,:] = np.array(train_R2s_i);
                train_KLs[check_it,:] = np.array(train_KLs_i);
                test_R2s[check_it,:] = np.array(test_R2s_i);
                test_KLs[check_it,:] = np.array(test_KLs_i);
                
                mean_train_elbo = np.mean(train_costs_i);
                mean_train_R2 = np.mean(train_R2s_i);
                mean_train_KL = np.mean(train_KLs_i);

                mean_test_elbo = np.mean(test_costs_i);
                mean_test_R2 = np.mean(test_R2s_i);
                mean_test_KL = np.mean(test_KLs_i);
                                
                print('cost', cost_i);
                print('train elbo: %f' % mean_train_elbo);
                print('train R2: %f' % mean_train_R2);
                if (not (exp_fam in ['prp_tn'])):
                    print('train KL: %f' % mean_train_KL);

                print('test elbo: %f' % mean_test_elbo);
                print('test R2: %f' % mean_test_R2);
                if (not (exp_fam in ['prp_tn'])):
                    print('test KL: %f' % mean_test_KL);
                #print('train R2: %.3f and train KL %.3f' % (mean_train_R2, mean_train_KL));

                if (dynamics):
                    np.savez(savedir + 'results.npz', As=As, sigma_epsilons=sigma_epsilons, autocov_targ=autocov_targ,  \
                                                      it=i, X=_X, check_rate=check_rate, eta=_eta, param_net_input=_param_net_input, params=eta_draw_params, \
                                                      train_elbos=train_elbos, test_elbos=test_elbos, \
                                                      train_R2s=train_R2s, test_R2s=test_R2s, \
                                                      train_KLs=train_KLs, test_KLs=test_KLs, final_cost=cost_i);
                else:
                    np.savez(savedir + 'results.npz', it=i, check_rate=check_rate, \
                                                      X=_X, eta=_eta, param_net_input=_param_net_input, params=eta_draw_params, \
                                                      train_elbos=train_elbos, test_elbos=test_elbos, \
                                                      train_R2s=train_R2s, test_R2s=test_R2s, \
                                                      train_KLs=train_KLs, test_KLs=test_KLs, final_cost=cost_i);

                check_it += 1;
            sys.stdout.flush();
            i += 1;

    #return _X, train_R2s, train_KLs, i;
    return _X, train_KLs, i;

