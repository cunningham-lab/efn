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
from efn_util import connect_flow, construct_flow, setup_IO, construct_param_network, \
                     cost_fn, check_convergence, \
                     setup_param_logging, count_params, get_param_network_hyperparams

def train_efn(family, flow_dict, param_net_input_type, cost_type, K, M, \
              stochastic_eta, give_hint=False, lr_order=-3, random_seed=0, \
              max_iters=10000, check_rate=200):
    batch_norm = False;
    dropout = False;
    upl_tau = None;
    upl_shape = 'linear';
    T = 1; # let's generalize to processes later (not within scope of NIPS submission)
    K_test_fac = 1;
    wsize = 50;
    delta_thresh = 1e-10;


    D_Z, num_suff_stats, num_param_net_inputs, num_T_x_inputs = family.get_efn_dims(param_net_input_type, give_hint);

    # set number of layers in the parameter network
    L = max(int(np.ceil(np.sqrt(D_Z))), 4);  # we use at least four layers

    # good practice
    tf.reset_default_graph();
    # seed RNGs
    tf.set_random_seed(random_seed);
    np.random.seed(random_seed);

    flow_layers, Z0, Z_AR, base_log_p_z, P, num_zi, num_theta_params, num_dyn_param_vals = construct_flow(flow_dict, D_Z, T);
    flow_layers, num_theta_params = family.map_to_support(flow_layers, num_theta_params);
    Z0_shape = tf.shape(Z0);
    batch_size = tf.multiply(Z0_shape[0], Z0_shape[1]);

    dynamics = P > 0;

    n = K*M;

    # optimization hyperparameters
    opt_method = 'adam';
    lr = 10**lr_order
    # save tensorboard summary in intervals
    tb_save_every = 50;
    model_save_every = max_iters-1;
    tb_save_params = True;

    eta = tf.placeholder(tf.float64, shape=(None, num_suff_stats));
    param_net_input = tf.placeholder(tf.float64, shape=(None, num_param_net_inputs));
    T_x_input = tf.placeholder(tf.float64, shape=(None, num_T_x_inputs));

    if (not stochastic_eta):
        # get etas based on constraint_id
        _eta, _param_net_input, _T_x_input, eta_draw_params = family.draw_etas(K, param_net_input_type, give_hint);
    _eta_tests = [];
    _param_net_input_tests = [];
    _T_x_input_tests = [];
    eta_test_draw_params = [];
    for i in range(K_test_fac):
        _eta_test, _param_net_input_test, _T_x_input_test, eta_test_draw_param = family.draw_etas(K, param_net_input_type, give_hint);
        _eta_tests.append(_eta_test);
        _param_net_input_tests.append(_param_net_input_test);
        _T_x_input_tests.append(_T_x_input_test);
        eta_test_draw_params.append(eta_test_draw_param);
           
    param_net_hps = get_param_network_hyperparams(L, num_param_net_inputs, num_theta_params, upl_tau, upl_shape);

    savedir = setup_IO(family, 'EFN', param_net_input_type, K, M, flow_dict, \
                       param_net_hps, stochastic_eta, give_hint, random_seed);
    if not os.path.exists(savedir):
        print('Making directory %s' % savedir );
        os.makedirs(savedir);

    # construct the parameter network
    theta = construct_param_network(param_net_input, K, flow_layers, param_net_hps);

    # connect time-invariant flow
    Z, sum_log_det_jacobian, Z_by_layer = connect_flow(Z_AR, flow_layers, theta);


    log_p_zs = base_log_p_z - sum_log_det_jacobian;

    # generative model is fully specified
    all_params = tf.trainable_variables();
    nparams = len(all_params);

    X = Z; # [n,D,T] 
    # set up the constraint computation
    T_x = family.compute_suff_stats(X, Z_by_layer, T_x_input);
    log_h_x = family.compute_log_base_measure(X);

    # exponential family optimization
    cost, costs, R2s = cost_fn(eta, log_p_zs, T_x, log_h_x, K, cost_type)
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
    tf.add_to_collection('T_x_input', T_x_input);
    saver = tf.train.Saver();

    # tensorboard logging
    summary_writer = tf.summary.FileWriter(savedir);
    tf.summary.scalar('cost', cost);

    if (tb_save_params):
        setup_param_logging(all_params);

    summary_op = tf.summary.merge_all();

    opt_compress_fac = 16;
    array_init_len = int(np.ceil(max_iters/opt_compress_fac));
    if (dynamics):
        As = np.zeros((array_init_len, K, D_Z, D_Z));
        sigma_epsilons = np.zeros((array_init_len,D_Z));
    array_cur_len = array_init_len;

    num_diagnostic_checks = (max_iters // check_rate) + 1;
    train_elbos = np.zeros((num_diagnostic_checks, K));
    test_elbos = np.zeros((num_diagnostic_checks, K_test_fac*K));
    train_R2s = np.zeros((num_diagnostic_checks, K));
    test_R2s = np.zeros((num_diagnostic_checks, K_test_fac*K));
    train_KLs = np.zeros((num_diagnostic_checks, K));
    test_KLs = np.zeros((num_diagnostic_checks, K_test_fac*K));
    check_it = 0;
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer();
        sess.run(init_op);

        z_i = np.random.normal(np.zeros((K, M, D_Z, num_zi)), 1.0);
        if (stochastic_eta):
            _eta, _param_net_input, _T_x_input, eta_draw_params = family.draw_etas(K, param_net_input_type, give_hint);
        feed_dict = {Z0:z_i, eta:_eta, param_net_input:_param_net_input, T_x_input:_T_x_input};

        cost_i, summary = \
            sess.run([cost, summary_op], feed_dict);

        if (dynamics):
            A_i, _sigma_epsilon_i = sess.run([A, sigma_eps]);
            As[0,:,:,:] = A_i;
            sigma_epsilons[0,:] = _sigma_epsilon_i[:,0];

        # compute R^2, KL, and elbo for training set
        z_i = np.random.normal(np.zeros((K, int(1e3), D_Z, num_zi)), 1.0);
        feed_dict_train = {Z0:z_i, eta:_eta, param_net_input:_param_net_input, T_x_input:_T_x_input};
        train_costs_i, train_R2s_i, train_KLs_i = family.batch_diagnostics(K, sess, feed_dict_train, X, log_p_zs, costs, R2s, eta_draw_params);
        train_elbos[check_it,:] = np.array(train_costs_i);
        train_R2s[check_it,:] = np.array(train_R2s_i);
        train_KLs[check_it,:] = np.array(train_KLs_i);

        # compute R^2, KL, and elbo for static test set
        for j in range(K_test_fac):
            z_i_test = np.random.normal(np.zeros((K, int(1e3), D_Z, num_zi)), 1.0);
            feed_dict_test = {Z0:z_i_test, eta:_eta_tests[j], param_net_input:_param_net_input_tests[j], T_x_input:_T_x_input_tests[j]};
            test_costs_i, test_R2s_i, test_KLs_i = family.batch_diagnostics(K, sess, feed_dict_test, X, log_p_zs, costs, R2s, eta_test_draw_params[j]);

            test_elbos[check_it,(j*K):((j+1)*K)] = np.array(test_costs_i);
            test_R2s[check_it,(j*K):((j+1)*K)] = np.array(test_R2s_i);
            test_KLs[check_it,(j*K):((j+1)*K)] = np.array(test_KLs_i);

        check_it += 1;

        optimizer = tf.train.AdamOptimizer(learning_rate=lr);

        train_step = optimizer.apply_gradients(grads_and_vars);
        
        init_op = tf.global_variables_initializer();
        sess.run(init_op);

        i = 1;

        print('starting opt');
        while (i < max_iters):
            z_i = np.random.normal(np.zeros((K, M, D_Z, num_zi)), 1.0);
            if (stochastic_eta): 
                _eta, _param_net_input, _T_x_input, eta_draw_params = family.draw_etas(K, param_net_input_type, give_hint);

            feed_dict = {Z0:z_i, eta:_eta, param_net_input:_param_net_input, T_x_input:_T_x_input};

            if (np.mod(i, check_rate)==0):
                start_time = time.time();
            ts, cost_i, summary = \
                sess.run([train_step, cost, summary_op], feed_dict);
            if (np.mod(i, check_rate)==0):
                end_time = time.time();
                print('iter %d took %f seconds' % (i+1, end_time-start_time));
                
            if (dynamics):
                A_i, _sigma_epsilon_i = sess.run([A, sigma_eps]);
                As[i,:,:] = A_i;
                sigma_epsilons[i,:] = _sigma_epsilon_i[:,0];

            if (np.mod(i,tb_save_every)==0):
                summary_writer.add_summary(summary, i);

            if (np.mod(i,model_save_every)==0):
                saver.save(sess, savedir + 'model');

            if (np.mod(i+1, check_rate)==0):
                print(42*'*');
                print('it = %d ' % (i+1));
                start_time = time.time();
                
                # compute R^2, KL, and elbo for training set
                z_i = np.random.normal(np.zeros((K, int(1e3), D_Z, num_zi)), 1.0);
                feed_dict_train = {Z0:z_i, eta:_eta, param_net_input:_param_net_input, T_x_input:_T_x_input};
                train_costs_i, train_R2s_i, train_KLs_i = family.batch_diagnostics(K, sess, feed_dict_train, X, log_p_zs, costs, R2s, eta_draw_params);
                train_elbos[check_it,:] = np.array(train_costs_i);
                train_R2s[check_it,:] = np.array(train_R2s_i);
                train_KLs[check_it,:] = np.array(train_KLs_i);
                mean_train_elbo = np.mean(train_costs_i);
                mean_train_R2 = np.mean(train_R2s_i);
                mean_train_KL = np.mean(train_KLs_i);

                # compute R^2, KL, and elbo for static test set
                for j in range(K_test_fac):
                    z_i_test = np.random.normal(np.zeros((K, int(1e3), D_Z, num_zi)), 1.0);
                    feed_dict_test = {Z0:z_i_test, eta:_eta_tests[j], param_net_input:_param_net_input_tests[j], T_x_input:_T_x_input_tests[j]};
                    test_costs_i, test_R2s_i, test_KLs_i = family.batch_diagnostics(K, sess, feed_dict_test, X, log_p_zs, costs, R2s, eta_test_draw_params[j]);

                    test_elbos[check_it,(j*K):((j+1)*K)] = np.array(test_costs_i);
                    test_R2s[check_it,(j*K):((j+1)*K)] = np.array(test_R2s_i);
                    test_KLs[check_it,(j*K):((j+1)*K)] = np.array(test_KLs_i);
                mean_test_elbo = np.mean(test_elbos[check_it,:]);
                mean_test_R2 = np.mean(test_R2s[check_it,:]);
                mean_test_KL = np.mean(test_KLs[check_it,:]);
                                
                print('cost', cost_i);
                print('train elbo: %f' % mean_train_elbo);
                print('train R2: %f' % mean_train_R2);
                if (family.name in ['dirichlet', 'normal', 'inv_wishart']):
                    print('train KL: %f' % mean_train_KL);

                print('test elbo: %f' % mean_test_elbo);
                print('test R2: %f' % mean_test_R2);
                if (family.name in ['dirichlet', 'normal', 'inv_wishart']):
                    print('test KL: %f' % mean_test_KL);

                _X = sess.run(X, feed_dict);
                if (dynamics):
                    np.savez(savedir + 'results.npz', As=As, sigma_epsilons=sigma_epsilons, autocov_targ=autocov_targ,  \
                                                      it=i, X=_X, check_rate=check_rate, eta=_eta, param_net_input=_param_net_input, params=eta_draw_params, \
                                                      T_x_input=_T_x_input, \
                                                      train_elbos=train_elbos, test_elbos=test_elbos, \
                                                      train_R2s=train_R2s, test_R2s=test_R2s, \
                                                      train_KLs=train_KLs, test_KLs=test_KLs, final_cost=cost_i);
                else:
                    np.savez(savedir + 'results.npz', it=i, check_rate=check_rate, \
                                                      X=_X, eta=_eta, param_net_input=_param_net_input, params=eta_draw_params, \
                                                      T_x_input=_T_x_input, \
                                                      train_elbos=train_elbos, test_elbos=test_elbos, \
                                                      train_R2s=train_R2s, test_R2s=test_R2s, \
                                                      train_KLs=train_KLs, test_KLs=test_KLs, final_cost=cost_i);

                if (check_it >= 2*wsize - 1 and (np.mod(i+1, wsize*check_rate)==0)):
                    last_mean_test_R2 = np.mean(test_R2s[(check_it-(2*wsize)+1):(check_it-wsize+1),:]);
                    cur_mean_test_R2 = np.mean(test_R2s[(check_it-wsize+1):(check_it+1),:]);
                    delta_R2 = (cur_mean_test_R2 - last_mean_test_R2) / last_mean_test_R2;

                    last_mean_test_elbo = np.mean(test_elbos[(check_it-(2*wsize)+1):(check_it-wsize+1),:]);
                    cur_mean_test_elbo = np.mean(test_elbos[(check_it-wsize+1):(check_it+1),:]);
                    delta_elbo = (last_mean_test_elbo - cur_mean_test_elbo) / last_mean_test_elbo;
                    if (delta_elbo < delta_thresh and delta_R2 < delta_thresh):
                        print('quitting opt:');
                        print('delta (elbo, r2) = (%f, %f) < %f' % (delta_elbo, delta_R2, delta_thresh));
                    else:
                        print('continue learning');
                        print('delta_elbo = %f' % delta_elbo);
                        print('delta_r2 = %f' % delta_R2);

                check_it += 1;
            sys.stdout.flush();
            i += 1;

    #return _X, train_R2s, train_KLs, i;
    return _X, train_KLs, i;

