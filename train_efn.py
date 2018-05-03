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
                      memory_extension, setup_param_logging, count_params, \
                      theta_network_hyperparams
from tensorflow.python import debug as tf_debug

def train_efn(exp_fam, D, flow_id, cost_type, K_eta, M_eta, stochastic_eta, \
              L_theta, batch_norm=False, dropout=False, lr_order=-3, random_seed=0, max_iters=10000, check_rate=200):
    T = 1; # let's generalize to processes later (not within scope of NIPS submission)
    stop_early = False;
    cost_grad_lag = 100;
    pthresh = 0.1;

    # seed RNGs
    np.random.seed(random_seed);
    tf.set_random_seed(random_seed);

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

    flow_layers, Z0, Z_AR, base_log_p_z, P, num_zi, num_theta_params, num_dyn_param_vals = construct_flow(flow_id, D_Z, T);
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

    eta = tf.placeholder(tf.float64, shape=(None, ncons));

    if (not stochastic_eta):
        # get etas based on constraint_id
        _eta, eta_draw_params = drawEtas(exp_fam, D_Z, K_eta);
        _eta_test, eta_test_draw_params = drawEtas(exp_fam, D_Z, K_eta);

    theta_nn_hps = theta_network_hyperparams(L_theta, ncons, num_theta_params);

    savedir = setup_IO(exp_fam, K_eta, M_eta, D, flow_id, theta_nn_hps, stochastic_eta, random_seed);
    print(random_seed, savedir);

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
    cost, all_costs, R2s = cost_fn(eta, log_p_zs, Tx, Bx, K_eta, cost_type)
    cost_grad = tf.gradients(cost, all_params);
    all_cost_grads = [];
    for i in range(K_eta):
        print('computing eta-grad %d' % (i+1));
        all_cost_grads = tf.gradients(all_costs[i], all_params);
    some_func = 0.0;
    for i in range(len(all_params)):
        some_func += tf.reduce_sum(all_params[i]);
    some_func_grads = tf.gradients(some_func, all_params);

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
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess);
        init_op = tf.global_variables_initializer();
        sess.run(init_op);

        z_i = np.random.normal(np.zeros((K_eta, M_eta, D_Z, num_zi)), 1.0);
        if (stochastic_eta):
            _eta, eta_draw_params = drawEtas(exp_fam, D_Z, K_eta);
            _eta_test, eta_test_draw_params = drawEtas(exp_fam, D_Z, K_eta);
        feed_dict = {Z0:z_i, eta:_eta};

        #cost_i, _cost_grads, _X, _y, _Tx, summary = \
        #    sess.run([cost, cost_grad, X, log_p_zs, Tx, summary_op], feed_dict);
        cost_i, _X, _y, _Tx, summary = \
            sess.run([cost, X, log_p_zs, Tx, summary_op], feed_dict);

        if (dynamics):
            A_i, _sigma_epsilon_i = sess.run([A, sigma_eps]);
            As[0,:,:,:] = A_i;
            sigma_epsilons[0,:] = _sigma_epsilon_i[:,0];

        #if (stop_early):
        #    log_grads(_cost_grads, cost_grad_vals, 0);

        optimizer = tf.train.AdamOptimizer(learning_rate=lr);

        train_step = optimizer.apply_gradients(grads_and_vars);
        #check_op = tf.add_check_numerics_ops();
        
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

            feed_dict = {Z0:z_i, eta:_eta};
            _cost, _all_costs, _cost_grads, _all_cost_grads, _X, _base_log_p_z, _sum_log_det_jacobian = sess.run([cost, all_costs, cost_grad, all_cost_grads, X, base_log_p_z, sum_log_det_jacobian], feed_dict);
            _theta, _all_params = sess.run([theta, all_params], feed_dict);
            _some_func, _some_func_grads = sess.run([some_func, some_func_grads]);
            print(42*'*');
            print('it = %d ' % (i+1));



            print('some func', _some_func);
            #print('before some func grads len', len(_some_func_grads));
            #for ii in range(len(_some_func_grads)):
            #    _some_func_grads_i = _some_func_grads[ii];
            #    for jj in range(len(_some_func_grads_i)):
            #        num_nans = np.sum(np.isnan(_some_func_grads_i[jj]));
            #        if (num_nans > 0):
            #            print('before some func grads', ii, jj, '%d/%d nans' % (num_nans, np.prod(_some_func_grads_i[jj].shape)));
            #            break;



            print('before cost', _cost);
            print('before all K costs', _all_costs);
            plt.figure();
            plt.hist(_all_costs);
            plt.show();
            print('before X nans', np.sum(np.isnan(_X)));
            print('before X infs', np.sum(np.isinf(_X)));
            print('before params len', len(_all_params));
            flag = False;
            for ii in range(len(_all_params)):
                _all_params_i = _all_params[ii];
                for jj in range(len(_all_params_i)):
                    num_nans = np.sum(np.isnan(_all_params_i[jj]));
                    num_infs = np.sum(np.isinf(_all_params_i[jj]));
                    if (num_nans > 0):
                        print('param', ii, jj, num_nans, '%d/%d nans' % (num_nans, np.prod(_all_params_i[jj].shape)));
                        flag = True;
                        break;
                    if (num_infs > 0):
                        print('param', ii, jj, num_infs, '%d/%d infs' % (num_infs, np.prod(_all_params_i[jj].shape)));
                if (flag):
                    break;
            print('before grads len', len(_cost_grads));
            flag = False;
            for ii in range(len(_cost_grads)):
                _cost_grad_i = _cost_grads[ii];
                for jj in range(len(_cost_grad_i)):
                    num_nans = np.sum(np.isnan(_cost_grad_i[jj]));
                    num_infs = np.sum(np.isinf(_cost_grad_i[jj]));
                    if (num_nans > 0):
                        print('before grad', ii, jj, '%d/%d nans' % (num_nans, np.prod(_cost_grad_i[jj].shape)));   
                        flag = True;
                        break;
                    if (num_infs > 0):
                        print('before grad', ii, jj, '%d/%d infs' % (num_infs, np.prod(_cost_grad_i[jj].shape)));
                if (flag):
                    plt.figure();
                    plt.hist(np.reshape(_base_log_p_z, (K_eta*M_eta,), 100));
                    plt.title('dist of z_is');
                    plt.show();

                    plt.figure();
                    plt.hist(np.reshape(_sum_log_det_jacobian, (K_eta*M_eta,), 100));
                    plt.title('dist of sum log det jac');
                    plt.show();
                    break;

            for k in range(K_eta):
                _cost_grad_k = _all_cost_grads[k];
                flag = False;
                for ii in range(len(_cost_grad_k)):
                    _cost_grad_i = _cost_grad_k[ii];
                    num_nans = np.sum(np.isnan(_cost_grad_i));
                    num_infs = np.sum(np.isinf(_cost_grad_i));
                    if (num_nans > 0):
                        print('before eta-grad %d', ii, '%d/%d nans' % (k, num_nans, np.prod(_cost_grad_i.shape)));   
                        flag = True;
                        break;
                    if (num_infs > 0):
                        print('before eta-grad %d', ii, '%d/%d infs' % (k, num_infs, np.prod(_cost_grad_i.shape)));
                    if (flag):
                        break;
            #start_time = time.time();
            #ts, cost_i, _X, _cost_grads, _log_p_zs, _Tx, summary = \
            #    sess.run([train_step, cost, X, cost_grad, log_p_zs, Tx, summary_op], feed_dict);
            ts, cost_i, _X, _log_p_zs, _Tx, summary = \
                sess.run([train_step, cost, X, log_p_zs, Tx, summary_op], feed_dict);
            #end_time = time.time();
            #end_time = time.time();
            #print('iter %d took %f seconds' % (i, end_time-start_time));
            _theta, _all_params = sess.run([theta, all_params], feed_dict);
            print('X shape', _X.shape);
            print('X', np.sum(np.isnan(_X)), 'nans');
            print('X', np.sum(np.isinf(_X)), 'infs');

            flag = False;
            print('after params len', len(_all_params));
            for ii in range(len(_all_params)):
                _all_params_i = _all_params[ii];
                for jj in range(len(_all_params_i)):
                    num_nans = np.sum(np.isnan(_all_params_i[jj]));
                    num_infs = np.sum(np.isinf(_all_params_i[jj]));
                    if (num_nans > 0):
                        flag = True;
                        print('param', ii, jj, num_nans, '%d/%d nans' % (num_nans, np.prod(_all_params_i[jj].shape)));
                        break;
                    if (num_infs > 0):
                        print('param', ii, jj, num_infs, '%d/%d infs' % (num_infs, np.prod(_all_params_i[jj].shape)));
                if (flag):
                    break;

            print('after grads len', len(_cost_grads));
            flag = False;
            for ii in range(len(_cost_grads)):
                _cost_grad_i = _cost_grads[ii];
                for jj in range(len(_cost_grad_i)):
                    num_nans = np.sum(np.isnan(_cost_grad_i[jj]));
                    num_infs = np.sum(np.isinf(_cost_grad_i[jj]));
                    if (num_nans > 0):
                        print('grad', ii, jj, num_nans, '%d/%d nans' % (num_nans, np.prod(_cost_grad_i[jj].shape)));
                        flag = True;
                        break;
                    if (num_infs > 0):
                        print('grad', ii, jj, num_infs, '%d/%d infs' % (num_infs, np.prod(_cost_grad_i[jj].shape)))
                if (flag):
                    break;

            #for i in range(len(_cost_grads)):
                #print(_cost_grads[i].shape);
            print('after _theta len', len(_theta));
            flag = False;
            for ii in range(len(_theta)):
                theta_i = _theta[ii];
                for jj in range(len(theta_i)):
                    num_nans = np.sum(np.isnan(theta_i[jj]));
                    if (num_nans > 0):
                        print('theta', ii, jj,num_nans, '%d/%d nans' % (num_nans, np.prod(theta_i[jj].shape)));
                        flag = True;
                        break;
                if (flag):
                    break;
            if (flag):
                exit();

                
            if (dynamics):
                A_i, _sigma_epsilon_i = sess.run([A, sigma_eps]);
                As[i,:,:] = A_i;
                sigma_epsilons[i,:] = _sigma_epsilon_i[:,0];

            #f (stop_early):
            #    log_grads(_cost_grads, cost_grad_vals, i);

            if (np.mod(i,tb_save_every)==0):
                summary_writer.add_summary(summary, i);

            if (np.mod(i+1, check_rate)==0):
                start_time = time.time();
                #if (stop_early):
                #    has_converged = check_convergence([cost_grad_vals], i, cost_grad_lag, pthresh, criteria='grad_mean_ttest');
                
                # compute R^2 and KL for training and batch
                z_i = np.random.normal(np.zeros((K_eta, int(1e3), D_Z, num_zi)), 1.0);
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
                                
                print('cost', cost_i);
                print('train R2: %f' % mean_train_R2);
                print('train KL: %f' % mean_train_KL);
                print('test R2: %f' % mean_test_R2);
                print('test KL: %f' % mean_test_KL);
                #print('train R2: %.3f and train KL %.3f' % (mean_train_R2, mean_train_KL));

                if (dynamics):
                    np.savez(savedir + 'results.npz', As=As, sigma_epsilons=sigma_epsilons, autocov_targ=autocov_targ,  \
                                                      it=i, X=_X, check_rate=check_rate, eta=_eta, params=eta_draw_params, \
                                                      train_R2s=train_R2s, test_R2s=test_R2s, \
                                                      train_KLs=train_KLs, test_KLs=test_KLs);
                else:
                    np.savez(savedir + 'results.npz', it=i, check_rate=check_rate, \
                                                      X=_X, eta=_eta, params=eta_draw_params, \
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
