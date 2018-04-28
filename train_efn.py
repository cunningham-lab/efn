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
                      setup_IO, construct_theta_network, log_grads, \
                      approxKL, drawEtas, checkH, declare_theta, cost_fn, \
                      computeLogBaseMeasure, check_convergence, batch_diagnostics, \
                      memory_extension, setup_param_logging, count_params

def train_efn(exp_fam, D, flow_id, cost_type, K_eta, M_eta, stochastic_eta, \
              theta_nn_hps, lr_order=-3, random_seed=0, max_iters=10000, check_rate=200):
    T = 1; # let's generalize to processes later (not within scope of NIPS submission)
    stop_early = False;
    cost_grad_lag = 100;
    pthresh = 0.1;

    if (exp_fam == 'dirichlet'):
        D_Z = D-1;
        ncons = D_Z+1;
    elif (exp_fam == 'normal'):
        D_Z = D;
        #ncons = int(D_Z+D_Z*(D_Z+1)/2);
        ncons = int(D + D**2);
    elif (exp_fam == 'inv_wishart'):
        sqrtD = int(np.sqrt(D));
        D_Z = int(sqrtD*(sqrtD+1)/2);
        ncons = D + 1;

    # good practice
    tf.reset_default_graph();

    flow_layers, Z0, Z_AR, base_log_p_z, P, num_zi, num_dyn_param_vals = construct_flow(flow_id, D_Z, T);
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
    tb_save_params = False;

    # seed RNGs
    np.random.seed(1);
    tf.set_random_seed(random_seed);

    savedir = setup_IO(exp_fam, K_eta, D, flow_id, theta_nn_hps, stochastic_eta, random_seed);
    eta = tf.placeholder(tf.float64, shape=(None, ncons));

    if (not stochastic_eta):
        # get etas based on constraint_id
        _eta, eta_draw_params = drawEtas(exp_fam, D_Z, K_eta);
        _eta_test, eta_test_draw_params = drawEtas(exp_fam, D_Z, K_eta);
        print(_eta);

    # construct the parameter network
    theta = construct_theta_network(eta, K_eta, flow_layers, theta_nn_hps);
    #theta = declare_theta(flow_layers);

    # connect time-invariant flow
    Z, sum_log_det_jacobian = connect_flow(Z_AR, flow_layers, theta, exp_fam);
    log_p_zs = base_log_p_z - sum_log_det_jacobian;

    # generative model is fully specified
    all_params = tf.trainable_variables();
    nparams = len(all_params);

    X = Z; # [n,D,T] 
    # set up the constraint computation
    Tx = computeMoments(X, exp_fam, D, T);
    Bx = computeLogBaseMeasure(X, exp_fam, D, T);

    # exponential family optimization
    cost, R2s = cost_fn(eta, log_p_zs, Tx, Bx, K_eta, cost_type)
    cost_grad = tf.gradients(cost, all_params);

    grads_and_vars = [];
    for i in range(len(all_params)):
        grads_and_vars.append((cost_grad[i], all_params[i]));

    # set optimization hyperparameters
    saver = tf.train.Saver();
    tf.add_to_collection('Z0', Z0);
    tf.add_to_collection('X', X);

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
            _eta, eta_draw_params = drawEtas(exp_fam, D_Z, K_eta);
        feed_dict = {Z0:z_i, eta:_eta};

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
        
        init_op = tf.global_variables_initializer();
        sess.run(init_op);

        i = 1;

        has_converged = False;
        while (i < max_iters):
            if (stop_early and i == array_cur_len):
                memory_extension(cost_grad_vals, array_cur_len);
                array_cur_len = 2*array_cur_len;

            z_i = np.random.normal(np.zeros((K_eta, M_eta, D_Z, num_zi)), 1.0);
            if (stochastic_eta): 
                _eta, eta_draw_params = drawEtas(exp_fam, D_Z, K_eta);
                _eta_test, eta_test_draw_params = drawEtas(exp_fam, D_Z, K_eta);

            feed_dict = {Z0:z_i, eta:_eta};


            start_time = time.time();
            ts, cost_i, _X, _cost_grads, _log_p_zs, _Tx, summary = \
                sess.run([train_step, cost, X, cost_grad, log_p_zs, Tx, summary_op], feed_dict);
            end_time = time.time();
            print('iter %d took %f seconds' % (i, end_time-start_time));
                
            if (dynamics):
                A_i, _sigma_epsilon_i = sess.run([A, sigma_eps]);
                As[i,:,:] = A_i;
                sigma_epsilons[i,:] = _sigma_epsilon_i[:,0];

            if (stop_early):
                log_grads(_cost_grads, cost_grad_vals, i);

            if (np.mod(i,tb_save_every)==0):
                summary_writer.add_summary(summary, i);

            if (np.mod(i+1, check_rate)==0):
                start_time = time.time();
                if (stop_early):
                    has_converged = check_convergence([cost_grad_vals], i, cost_grad_lag, pthresh, criteria='grad_mean_ttest');
                
                # compute R^2 and KL for training and batch
                z_i = np.random.normal(np.zeros((K_eta, int(1e4), D_Z, num_zi)), 1.0);
                feed_dict_train = {Z0:z_i, eta:_eta};
                feed_dict_test = {Z0:z_i, eta:_eta_test};

                print('Training');
                train_R2s_i, train_KLs_i = batch_diagnostics(exp_fam, K_eta, sess, feed_dict_train, X, log_p_zs, R2s, eta_draw_params);
                print('Testing');
                test_R2s_i, test_KLs_i = batch_diagnostics(exp_fam, K_eta, sess, feed_dict_test, X, log_p_zs, R2s, eta_test_draw_params);
                end_time = time.time();
                print('check diagnostics processes took: %f seconds' % (end_time-start_time));

                train_R2s[check_it,:] = np.array(train_R2s_i);
                train_KLs[check_it,:] = np.array(train_KLs_i);
                test_R2s[check_it,:] = np.array(test_R2s_i);
                test_KLs[check_it,:] = np.array(test_KLs_i);

                mean_train_R2 = np.mean(train_R2s_i);
                mean_train_KL = np.mean(train_KLs_i);
                mean_test_R2 = np.mean(test_R2s_i);
                mean_test_KL = np.mean(test_KLs_i);
                                
                print(42*'*');
                print('it = %d ' % (i+1));
                print('cost', cost_i);
                print('train R2: %f' % mean_train_R2);
                print('train KL: %f' % mean_train_KL);
                print('test R2: %f' % mean_test_R2);
                print('test KL: %f' % mean_test_KL);
                #print('train R2: %.3f and train KL %.3f' % (mean_train_R2, mean_train_KL));

                if (dynamics):
                    np.savez(savedir + 'results.npz', As=As, sigma_epsilons=sigma_epsilons, autocov_targ=autocov_targ,  \
                                                      it=i, X=_X, check_rate=check_rate, \
                                                      train_R2s=train_R2s, test_R2s=test_R2s, \
                                                      train_KLs=train_KLs, test_KLs=test_KLs);
                else:
                    np.savez(savedir + 'results.npz', it=i, check_rate=check_rate, \
                                                      X=_X, \
                                                      train_R2s=train_R2s, test_R2s=test_R2s, \
                                                      train_KLs=train_KLs, test_KLs=test_KLs);
                
                check_it += 1;
            sys.stdout.flush();
            i += 1;

        # save all the hyperparams
        if not os.path.exists(savedir):
                print('Making directory %s' % savedir );
                os.makedirs(savedir);
        #saveParams(params, savedir);
        # save the model
        saver.save(sess, savedir + 'model');
    #return _X, train_R2s, train_KLs, i;
    return _X, train_KLs, i;

if __name__ == '__main__':    # parse command line parameters
    n_args = len(sys.argv);
    exp_fam = str(sys.argv[1]);
    D = int(sys.argv[2]); 
    flow_id = str(sys.argv[3]);
    cost_type = str(sys.argv[4]);
    K_eta = int(sys.argv[5]);
    M_eta = int(sys.argv[6]);
    stochastic_eta_input = sys.argv[7];
    L_theta = int(sys.argv[8]);
    upl_theta = int(sys.argv[9]);
    lr_order = float(sys.argv[10]);
    random_seed = int(sys.argv[11]);
    
    stochastic_eta = not (str(stochastic_eta_input) == 'False');
    theta_nn_hps = {'L':L_theta, 'upl':upl_theta};

    train_efn(exp_fam, flow_id, cost_type, L_theta, upl_theta, n, K_eta, \
                  theta_nn_hps, lr_order, random_seed);
