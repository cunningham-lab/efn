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
                     cost_fn, check_convergence, memory_extension, \
                     setup_param_logging, count_params, get_param_network_hyperparams, \
                     test_convergence

def train_efn(family, flow_dict, param_net_input_type, cost_type, K, M, \
              stochastic_eta, give_hint=False, lr_order=-3, dist_seed=0, random_seed=0, \
              max_iters=10000, check_rate=200, dir_str='general'):
    batch_norm = False;
    dropout = False;
    upl_tau = None;
    upl_shape = 'linear';
    T = 1; 
    wsize = 50;
    delta_thresh = 1e-10;
    min_iters = 50000;


    D_Z, num_suff_stats, num_param_net_inputs, num_T_x_inputs = family.get_efn_dims(param_net_input_type, give_hint);

    # set number of layers in the parameter network
    L = max(int(np.ceil(np.sqrt(D_Z))), 4);  # we use at least four layers

    # good practice
    tf.reset_default_graph();
    # seed RNGs
    tf.set_random_seed(random_seed);

    flow_layers, Z0, Z_AR, base_log_p_z, num_theta_params = construct_flow(flow_dict, D_Z, T, random_seed);
    flow_layers, num_theta_params = family.map_to_support(flow_layers, num_theta_params);
    Z0_shape = tf.shape(Z0);
    batch_size = tf.multiply(Z0_shape[0], Z0_shape[1]);

    n = K*M;

    # optimization hyperparameters
    opt_method = 'adam';
    lr = 10**lr_order
    # save tensorboard summary in intervals
    tb_save_every = 50;
    model_save_every = max_iters-1;
    tb_save_params = False;

    eta = tf.placeholder(tf.float64, shape=(None, num_suff_stats));
    param_net_input = tf.placeholder(tf.float64, shape=(None, num_param_net_inputs));
    T_x_input = tf.placeholder(tf.float64, shape=(None, num_T_x_inputs));

    np.random.seed(dist_seed);
    if (not stochastic_eta):
        # get etas based on constraint_id
        _eta, _param_net_input, _T_x_input, eta_draw_params = family.draw_etas(K, param_net_input_type, give_hint);
    _eta_tests = [];
    _param_net_input_tests = [];
    _T_x_input_tests = [];
    eta_test_draw_params = [];
    _eta_test, _param_net_input_test, _T_x_input_test, eta_test_draw_params = family.draw_etas(K, param_net_input_type, give_hint);
           
    param_net_hps = get_param_network_hyperparams(L, num_param_net_inputs, num_theta_params, upl_tau, upl_shape);
    dist_info = {'dist_seed':dist_seed};
    efn_str = 'EFN1' if (K ==1) else 'EFN';
    savedir = setup_IO(family, 'EFN', dir_str, param_net_input_type, K, M, flow_dict, \
                       param_net_hps, stochastic_eta, give_hint, random_seed, dist_info);

    if not os.path.exists(savedir):
        print('Making directory %s' % savedir );
        os.makedirs(savedir);

    # construct the parameter network
    theta = construct_param_network(param_net_input, K, flow_layers, param_net_hps, random_seed);
    #feed_dict = {param_net_input:_param_net_input};

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

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr);
    train_step = optimizer.minimize(cost);

    summary_op = tf.summary.merge_all();

    opt_compress_fac = 128;
    max_diagnostic_checks = (max_iters // check_rate) + 1;
    array_init_len = int(np.ceil(max_diagnostic_checks/opt_compress_fac));
    array_cur_len = array_init_len;
    print('array init len', array_init_len);

    train_elbos = np.zeros((array_init_len, K));
    test_elbos = np.zeros((array_init_len, K));
    train_R2s = np.zeros((array_init_len, K));
    test_R2s = np.zeros((array_init_len, K));
    train_KLs = np.zeros((array_init_len, K));
    test_KLs = np.zeros((array_init_len, K));
    check_it = 0;
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer();
        sess.run(init_op);
        z_i = np.random.normal(np.zeros((K, M, D_Z, T)), 1.0);
        if (stochastic_eta):
            _eta, _param_net_input, _T_x_input, eta_draw_params = family.draw_etas(K, param_net_input_type, give_hint);
        feed_dict = {Z0:z_i, eta:_eta, param_net_input:_param_net_input, T_x_input:_T_x_input};

        cost_i, summary = \
            sess.run([cost, summary_op], feed_dict);

        # compute R^2, KL, and elbo for training set
        z_i = np.random.normal(np.zeros((K, int(1e3), D_Z, T)), 1.0);
        feed_dict_train = {Z0:z_i, eta:_eta, param_net_input:_param_net_input, T_x_input:_T_x_input};
        train_costs_i, train_R2s_i, train_KLs_i = family.batch_diagnostics(K, sess, feed_dict_train, X, log_p_zs, costs, R2s, eta_draw_params);
        train_elbos[check_it,:] = np.array(train_costs_i);
        train_R2s[check_it,:] = np.array(train_R2s_i);
        train_KLs[check_it,:] = np.array(train_KLs_i);

        # compute R^2, KL, and elbo for static test set
        feed_dict_test = {Z0:z_i, eta:_eta_test, param_net_input:_param_net_input_test, T_x_input:_T_x_input_test};
        test_costs_i, test_R2s_i, test_KLs_i = family.batch_diagnostics(K, sess, feed_dict_test, X, log_p_zs, costs, R2s, eta_test_draw_params);

        test_elbos[check_it,:] = np.array(test_costs_i);
        test_R2s[check_it,:] = np.array(test_R2s_i);
        test_KLs[check_it,:] = np.array(test_KLs_i);

        check_it += 1;

        i = 1;

        print('starting opt');
        while (i < max_iters):
            z_i = np.random.normal(np.zeros((K, M, D_Z, T)), 1.0);
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

            if (np.mod(i,tb_save_every)==0):
                summary_writer.add_summary(summary, i);

            if (np.mod(i,model_save_every)==0):
                saver.save(sess, savedir + 'model');

            if (np.mod(i+1, check_rate)==0):
                print(42*'*');
                print(savedir);
                print('it = %d ' % (i+1));
                print('cost', cost_i);
                start_time = time.time();
                
                # compute R^2, KL, and elbo for training set
                z_i = np.random.normal(np.zeros((K, int(1e3), D_Z, T)), 1.0);
                feed_dict_train = {Z0:z_i, eta:_eta, param_net_input:_param_net_input, T_x_input:_T_x_input};

                train_costs_i, train_R2s_i, train_KLs_i = family.batch_diagnostics(K, sess, feed_dict_train, X, log_p_zs, costs, R2s, eta_draw_params);
                train_elbos[check_it,:] = np.array(train_costs_i);
                train_R2s[check_it,:] = np.array(train_R2s_i);
                train_KLs[check_it,:] = np.array(train_KLs_i);
                mean_train_elbo = np.mean(train_costs_i);
                mean_train_R2 = np.mean(train_R2s_i);
                mean_train_KL = np.mean(train_KLs_i);

                # compute R^2, KL, and elbo for static test set
                feed_dict_test = {Z0:z_i, eta:_eta_test, param_net_input:_param_net_input_test, T_x_input:_T_x_input_test};
                test_costs_i, test_R2s_i, test_KLs_i = family.batch_diagnostics(K, sess, feed_dict_test, X, log_p_zs, costs, R2s, eta_test_draw_params);

                test_elbos[check_it,:] = np.array(test_costs_i);
                test_R2s[check_it,:] = np.array(test_R2s_i);
                test_KLs[check_it,:] = np.array(test_KLs_i);

                mean_test_elbo = np.mean(test_elbos[check_it,:]);
                mean_test_R2 = np.mean(test_R2s[check_it,:]);
                mean_test_KL = np.mean(test_KLs[check_it,:]);
                                
                print('train elbo: %f' % mean_train_elbo);
                if (family.name in ['dirichlet', 'normal', 'inv_wishart']):
                    print('train KL: %f' % mean_train_KL);
                print('train R2: %f' % mean_train_R2);

                print('test elbo: %f' % mean_test_elbo);
                if (family.name in ['dirichlet', 'normal', 'inv_wishart']):
                    print('test KL: %f' % mean_test_KL);
                print('test R2: %f' % mean_test_R2);

                _X = sess.run(X, feed_dict);
                np.savez(savedir + 'results.npz', it=i, check_rate=check_rate, \
                                                  X=_X, eta=_eta, param_net_input=_param_net_input, params=eta_draw_params, \
                                                  T_x_input=_T_x_input, converged=False, \
                                                  train_elbos=train_elbos, test_elbos=test_elbos, \
                                                  train_R2s=train_R2s, test_R2s=test_R2s, \
                                                  train_KLs=train_KLs, test_KLs=test_KLs, final_cost=cost_i);

                if (check_it >= 2*wsize - 1):
                    mean_test_elbos = np.mean(test_elbos, 1);
                    if (i >= min_iters):
                        if (test_convergence(mean_test_elbos, check_it, wsize, delta_thresh)):
                            print('converged!');
                            break;

                check_it += 1;

                if (check_it == array_cur_len):
                    print('Extending log length from %d to %d' % (array_cur_len, 2*array_cur_len));
                    train_elbos, train_R2s, train_KLs, test_elbos, test_R2s, test_KLs = \
                        memory_extension(train_elbos, train_R2s, train_KLs, test_elbos, test_R2s, test_KLs, array_cur_len);
                    array_cur_len = 2*array_cur_len;
            sys.stdout.flush();
            i += 1;
        print('saving model before exitting');
        saver.save(sess, savedir + 'model');
    if (i < max_iters):
        np.savez(savedir + 'results.npz', it=i, check_rate=check_rate, \
                                      X=_X, eta=_eta, param_net_input=_param_net_input, params=eta_draw_params, \
                                      T_x_input=_T_x_input, converged=True, \
                                      train_elbos=train_elbos, test_elbos=test_elbos, \
                                      train_R2s=train_R2s, test_R2s=test_R2s, \
                                      train_KLs=train_KLs, test_KLs=test_KLs, final_cost=cost_i);


    return _X, train_KLs, i;

