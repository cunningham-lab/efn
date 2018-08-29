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
from efn_util import setup_IO, log_grads, cost_fn, test_convergence, \
                     memory_extension, setup_param_logging
from tf_util.tf_util import connect_flow, construct_flow, declare_theta, count_params

def train_nf(family, params, flow_dict, cost_type, M=1000, lr_order=-3, random_seed=0, \
             min_iters=100000, max_iters=1000000, check_rate=100, dir_str='general'):
    T = 1;
    wsize = 50;
    delta_thresh = 1e-10;
 
    D_Z, num_suff_stats, num_param_net_inputs, num_T_x_inputs = family.get_efn_dims('eta', False);

    # good practice
    tf.reset_default_graph();

    flow_layers, Z0, Z_AR, base_log_p_z, num_theta_params = construct_flow(flow_dict, D_Z, family.T);
    flow_layers, num_theta_params = family.map_to_support(flow_layers, num_theta_params);
    Z0_shape = tf.shape(Z0);
    batch_size = tf.multiply(Z0_shape[0], Z0_shape[1]);

    K = 1;
    n = K*M;

    # optimization hyperparameters
    lr = 10**lr_order
    # save tensorboard summary in intervals
    model_save_every = max_iters-1;
    tb_save_every = 50;
    tb_save_params = True;

    # seed RNGs
    np.random.seed(0);
    tf.set_random_seed(random_seed);

    dist_info = {'dist_seed':params['dist_seed']};
    savedir = setup_IO(family, 'NF1', dir_str, '', K, M, flow_dict, {}, False, False, random_seed, dist_info);
    eta = tf.placeholder(tf.float64, shape=(None, num_suff_stats), name='eta');
    T_x_input = tf.placeholder(tf.float64, shape=(None, num_T_x_inputs), name='T_x_input');

    print('params');
    print(params);
    _eta, _ = family.mu_to_eta(params, 'eta', False);
    _eta = np.expand_dims(_eta, 0);
    _T_x_input = family.mu_to_T_x_input(params);
    _T_x_input = np.expand_dims(_T_x_input, 0);

    _eta_test = _eta;
    _T_x_input_test = _T_x_input;

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
    tf.add_to_collection('log_p_zs', log_p_zs);
    tf.add_to_collection('T_x_input', T_x_input);
    saver = tf.train.Saver();

    # tensorboard logging
    summary_writer = tf.summary.FileWriter(savedir);
    for k in range(K):
        tf.summary.scalar('R2%d' % (k+1), R2s[k]);
    tf.summary.scalar('cost', cost);

    # log parameter values throughout optimization
    if (tb_save_params):
        setup_param_logging(all_params);

    summary_op = tf.summary.merge_all()

    opt_compress_fac = 16;
    array_init_len = int(np.ceil(max_iters/opt_compress_fac));
    array_cur_len = array_init_len;

    num_diagnostic_checks = (max_iters // check_rate) + 1;
    train_elbos = np.zeros((num_diagnostic_checks, K));
    test_elbos = np.zeros((num_diagnostic_checks, K));
    train_R2s = np.zeros((num_diagnostic_checks, K));
    test_R2s = np.zeros((num_diagnostic_checks, K));
    train_KLs = np.zeros((num_diagnostic_checks, K));
    test_KLs = np.zeros((num_diagnostic_checks, K));
    check_it = 0;
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer();
        sess.run(init_op);

        z_i = np.random.normal(np.zeros((K, M, D_Z, T)), 1.0);
        feed_dict = {Z0:z_i, eta:_eta, T_x_input:_T_x_input};

        cost_i, summary = \
            sess.run([cost, summary_op], feed_dict);

        z_i = np.random.normal(np.zeros((K, int(1e3), D_Z, T)), 1.0);
        feed_dict_train = {Z0:z_i, eta:_eta, T_x_input:_T_x_input};
        feed_dict_test = {Z0:z_i, eta:_eta_test, T_x_input:_T_x_input_test};
        train_elbos_i, train_R2s_i, train_KLs_i, train_X = family.batch_diagnostics(K, sess, feed_dict_train, X, log_p_zs, costs, R2s, [params], True);
        test_elbos_i, test_R2s_i, test_KLs_i, test_X = family.batch_diagnostics(K, sess, feed_dict_test, X, log_p_zs, costs, R2s, [params]);

        train_elbos[check_it,:] = np.array(train_elbos_i);
        test_elbos[check_it,:] = np.array(test_elbos_i);
        train_R2s[check_it,:] = np.array(train_R2s_i);
        train_KLs[check_it,:] = np.array(train_KLs_i);
        test_R2s[check_it,:] = np.array(test_R2s_i);
        test_KLs[check_it,:] = np.array(test_KLs_i);
        check_it += 1;

        optimizer = tf.train.AdamOptimizer(learning_rate=lr);

        train_step = optimizer.apply_gradients(grads_and_vars);
        
        init_op = tf.global_variables_initializer();
        sess.run(init_op);

        i = 1;

        has_converged = False;
        while (i < max_iters):
            z_i = np.random.normal(np.zeros((K, M, D_Z, T)), 1.0);
            feed_dict = {Z0:z_i, eta:_eta, T_x_input:_T_x_input};

            if (np.mod(i, check_rate)==0):
                start_time = time.time();
            ts, cost_i, _cost_grads, _R2s, _T_x, _log_h_x, summary = \
                    sess.run([train_step, cost, cost_grad, R2s, T_x, log_h_x, summary_op], feed_dict);

            if (np.mod(i, check_rate)==0):
                end_time = time.time();
                print('iter %d took %f seconds' % (i+1, end_time-start_time));

            if (np.mod(i,tb_save_every)==0):
                summary_writer.add_summary(summary, i);

            if (np.mod(i,model_save_every) == 0):
                # save all the hyperparams
                if not os.path.exists(savedir):
                    print('Making directory %s' % savedir );
                    os.makedirs(savedir);
                print('saving model at iter', i);
                saver.save(sess, savedir + 'model');

            if (np.mod(i+1, check_rate)==0):
                z_i = np.random.normal(np.zeros((K, int(1e3), D_Z, T)), 1.0);
                feed_dict_train = {Z0:z_i, eta:_eta, T_x_input:_T_x_input};
                feed_dict_test = {Z0:z_i, eta:_eta_test, T_x_input:_T_x_input_test};
                train_elbos_i, train_R2s_i, train_KLs_i, train_X = family.batch_diagnostics(K, sess, feed_dict_train, X, log_p_zs, costs, R2s, [params], True);
                test_elbos_i, test_R2s_i, test_KLs_i, test_X = family.batch_diagnostics(K, sess, feed_dict_test, X, log_p_zs, costs, R2s, [params]);

                train_elbos[check_it,:] = np.array(train_elbos_i);
                test_elbos[check_it,:] = np.array(test_elbos_i);
                train_R2s[check_it,:] = np.array(train_R2s_i);
                train_KLs[check_it,:] = np.array(train_KLs_i);
                test_R2s[check_it,:] = np.array(test_R2s_i);
                test_KLs[check_it,:] = np.array(test_KLs_i);
                mean_train_elbo = np.mean(train_elbos_i);
                mean_train_R2 = np.mean(train_R2s_i);
                mean_train_KL = np.mean(train_KLs_i);

                print(42*'*');
                print(savedir);
                print('it = %d ' % (i+1));
                print('cost = %f ' % cost_i);
                print('train elbo %.3f, train R2: %.3f, train KL %.3f' % (mean_train_elbo, mean_train_R2, mean_train_KL));
                
                np.savez(savedir + 'results.npz', it=i, X=train_X, eta=_eta, \
                                                  T_x_input=_T_x_input, params=params, check_rate=check_rate, \
                                                  train_elbos=train_elbos, test_elbos=test_elbos, \
                                                  train_R2s=train_R2s, test_R2s=test_R2s, \
                                                  train_KLs=train_KLs, test_KLs=test_KLs, \
                                                  converged=False, final_cos=cost_i);

                if (check_it >= 2*wsize - 1):
                    mean_test_elbos = np.mean(test_elbos, 1);
                    if (i >= min_iters):
                        if (test_convergence(mean_test_elbos, check_it, wsize, delta_thresh)):
                            print('converged!');
                            break;
                
                check_it += 1;

            sys.stdout.flush();
            i += 1;

        z_i = np.random.normal(np.zeros((K, M, D_Z, T)), 1.0);
        feed_dict = {Z0:z_i, eta:_eta, T_x_input:_T_x_input};
        _log_p_zs, _X = sess.run([log_p_zs, X], feed_dict);

    if (i < max_iters):
        np.savez(savedir + 'results.npz', it=i, X=train_X, eta=_eta, T_x_input=_T_x_input, params=params, check_rate=check_rate, \
                                      train_elbos=train_elbos, test_elbos=test_elbos, \
                                      train_R2s=train_R2s, test_R2s=test_R2s, \
                                      train_KLs=train_KLs, test_KLs=test_KLs, \
                                      converged=True, final_cos=cost_i);

    return _log_p_zs, train_X, train_R2s, train_KLs, i;
