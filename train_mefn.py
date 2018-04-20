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
from efn_util import MMD2u, PlanarFlowLayer, computeMoments, getEtas, \
                      latent_dynamics, connect_flow, construct_flow, \
                      setup_IO, normal_eta, \
                      approxKL, drawEtas, checkH, declare_theta, cost_fn, \
                      computeLogBaseMeasure, check_convergence

def train_mefn(exp_fam, params, flow_id, cost_type, M_eta=100, lr_order=-3, random_seed=0, check_rate=1000):
    T = 1; # let's generalize to processes later :P (not within scope of NIPS submission)
    cost_grad_lag = check_rate;
    pthresh = 0.1;
 
    D = params['D'];
    if (exp_fam == 'dirichlet'):
        D_Z = D-1;
    else:
        D_Z = D;
    if (exp_fam == 'normal'):
        ncons = D_Z+D_Z**2;
    elif (exp_fam == 'dirichlet'):
        ncons = D_Z+1;

    # good practice
    tf.reset_default_graph();

    flow_layers, Z0, Z_AR, base_log_p_z, P, num_zi, num_dyn_param_vals = construct_flow(flow_id, D_Z, T);
    K = tf.shape(Z0)[0];
    M = tf.shape(Z0)[1];
    batch_size = tf.multiply(K, M);
    dynamics = P > 0;

    K_eta = 1;
    n = K_eta*M_eta;

    # optimization hyperparameters
    max_iters = 50000;
    lr = 10**lr_order
    # save tensorboard summary in intervals
    tb_save_every = 1;
    tb_save_flow_param = True;
    np_save_flow_param = False;

    # seed RNGs
    np.random.seed(0);
    tf.set_random_seed(random_seed);

    savedir = setup_IO(exp_fam, D, flow_id, {}, False, random_seed);
    eta = tf.placeholder(tf.float64, shape=(None, ncons));

    if (exp_fam == 'normal'):
        mu_targ = params['mu'];
        Sigma_targ = params['Sigma'];
    elif (exp_fam == 'dirichlet'):
        alpha_targ = params['alpha'];

    if (exp_fam == 'normal'):
        _eta = np.transpose(normal_eta(mu_targ, Sigma_targ));
    elif (exp_fam == 'dirichlet'):
        _eta = np.expand_dims(alpha_targ, 0);
    _eta_test = _eta;

    # construct the parameter network
    L_flow = len(flow_layers);
    theta = declare_theta(flow_layers);

    # connect time-invariant flow
    Z, sum_log_det_jacobian, Z_pf, log_det_jac_list, input_to_log_abs_list = connect_flow(Z_AR, flow_layers, theta, exp_fam, K_eta, M_eta);
    log_p_zs = base_log_p_z - sum_log_det_jacobian;

    # generative model is fully specified
    all_params = tf.trainable_variables();
    nparams = len(all_params);

    X = Z; # [n,D,T] 
    X_cov = tf.div(tf.matmul(X, tf.transpose(X, [0, 1, 3, 2])), T); # this is [n x D x D]
    # set up the constraint computation
    Tx = computeMoments(X, exp_fam, D, T);
    Bx = computeLogBaseMeasure(X, exp_fam, D, T);
    # exponential family optimization
    cost, R2s = cost_fn(eta, log_p_zs, Tx, Bx, K_eta, cost_type)
    cost_grad = tf.gradients(cost, all_params);

    # create readout of Tx*eta
    eta_dummy = tf.expand_dims(eta[0,:],1);
    Txeta = tf.matmul(Tx[0,:,:], eta_dummy);

    # set optimization hyperparameters
    saver = tf.train.Saver();
    tf.add_to_collection('Z0', Z0);
    tf.add_to_collection('X', X);

    # tensorboard logging
    summary_writer = tf.summary.FileWriter(savedir);
    for k in range(K_eta):
        tf.summary.scalar('R2%d' % (k+1), R2s[k]);
    tf.summary.scalar('cost', cost);
    for k in range(L_flow):
        tf.summary.histogram('layer%d u^T phi' % (k+1), input_to_log_abs_list[k]);
        tf.summary.histogram('layer%d log det jacs' % (k+1), log_det_jac_list[k]);

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

    opt_compress_fac = 16;
    array_init_len = int(np.ceil(max_iters/opt_compress_fac));
    if (dynamics):
        As = np.zeros((array_init_len, K, D_Z, D_Z));
        sigma_epsilons = np.zeros((array_init_len,D_Z));
    #X_covs = np.zeros((array_init_len, D, D));
    flow_param_vals = np.zeros((array_init_len, num_flow_param_vals,));
    cost_grad_vals = np.zeros((array_init_len, num_dyn_param_vals + num_flow_param_vals));
    array_cur_len = array_init_len;

    num_diagnostic_checks = (max_iters // check_rate) + 1;
    train_R2s = np.zeros((num_diagnostic_checks, K_eta));
    test_R2s = np.zeros((num_diagnostic_checks, K_eta));
    train_KLs = np.zeros((num_diagnostic_checks, K_eta));
    test_KLs = np.zeros((num_diagnostic_checks, K_eta));
    check_it = 0;
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer();
        sess.run(init_op);

        print('D=%d, T=%d, P=%d, n=%d, lr=10^%.1f' % (D_Z, T, P, n, np.log10(lr)));
        grads_and_vars = [];
        for i in range(len(all_params)):
            grads_and_vars.append((cost_grad[i], all_params[i]));
        z_i = np.random.normal(np.zeros((K_eta, M_eta, D_Z, num_zi)), 1.0);
        feed_dict = {Z0:z_i, eta:_eta};
        if (dynamics):
            cost_i, A_i, X_cov_i, _sigma_epsilon_i, _flow_params, _cost_grads, summary = \
                    sess.run([cost, A, X_cov, sigma_eps, flow_params, cost_grad, summary_op], feed_dict);
            As[0,:,:,:] = A_i;
            sigma_epsilons[0,:] = _sigma_epsilon_i[:,0];
            #X_covs[0, :, :] = np.mean(X_cov_i, 0);
        else:
            cost_i, X_cov_i, _flow_params, _cost_grads, _X, _y, _Tx, summary = \
                    sess.run([cost, X_cov, flow_params, cost_grad, X, log_p_zs, Tx, summary_op], feed_dict);
            #X_covs[0, :, :] = np.mean(X_cov_i, 0);

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
        optimizer = tf.train.AdamOptimizer(learning_rate=lr);

        train_step = optimizer.apply_gradients(grads_and_vars);
        
        init_op = tf.global_variables_initializer();
        sess.run(init_op);

        # SGD iteration
        i = 1;

        #log_param_diff = log_param_diff_thresh + 1;
        has_converged = False;
        #while ((i < (cost_grad_lag)) or (not has_converged)): 
        convergence_it = 0;
        while (i < max_iters):
            if (i == array_cur_len):
                if (dynamics):
                    As = np.concatenate((As, np.zeros((array_cur_len, K, D_Z, D_Z))), axis=0);
                    sigma_epsilons = np.concatenate((sigma_epsilons, np.zeros((array_cur_len,D))), axis=0);
                #X_covs = np.concatenate((X_covs, np.zeros((array_cur_len, D, D))), axis=0);
                if (np_save_flow_param):
                    flow_param_vals = np.concatenate((flow_param_vals, np.zeros((array_cur_len, num_flow_param_vals))), axis=0);
                cost_grad_vals = np.concatenate((cost_grad_vals, np.zeros((array_cur_len, num_flow_param_vals))), axis=0);
                
                # double array lengths
                array_cur_len = 2*array_cur_len;

            z_i = np.random.normal(np.zeros((K_eta, M_eta, D_Z, num_zi)), 1.0);
            feed_dict = {Z0:z_i, eta:_eta};
            if (dynamics):
                ts, cost_i, A_i, X_cov_i, _sigma_epsilon_i, _X, _flow_params, _cost_grads, _R2s, summary = \
                    sess.run([train_step, cost, A, X_cov, sigma_eps, X, flow_params, cost_grad, R2s, summary_op], feed_dict);
                As[i,:,:] = A_i;
                sigma_epsilons[i,:] = _sigma_epsilon_i[:,0];
                #X_covs[i, :, :] = np.mean(X_cov_i, 0);
            else:
                ts, cost_i, X_cov_i, _X, _flow_params, _cost_grads, _R2s, _log_p_zs, _Tx, summary = \
                    sess.run([train_step, cost, X_cov, X, flow_params, cost_grad, R2s, log_p_zs, Tx, summary_op], feed_dict);
                #X_covs[i, :, :] = np.mean(X_cov_i, 0);

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

            if (np.mod(i, check_rate)==0):
                #has_converged = check_convergence([cost_grad_vals], i, cost_grad_lag, pthresh, criteria='grad_mean_ttest');
                has_converged = False;
                train_R2s[check_it,:] = _R2s;
                # compute KL
                training_KL = [];
                _y = np.expand_dims(_log_p_zs[0,:], 1);
                _X = _X[0, :, :, 0];
                _Z_pf, _input_to_log_abs, _log_det_jacs = sess.run([Z_pf, input_to_log_abs_list, log_det_jac_list] , feed_dict);
                training_KL.append(approxKL(_Z_pf[0,:,:,0], _y, _X, exp_fam, params, True));
                train_KLs[check_it,:] = training_KL;
                for f_ind in range(L_flow):
                    input_to_log_abs_i = _input_to_log_abs[f_ind];
                    log_det_jacs_i = _log_det_jacs[f_ind];
                    # compute best approx I can get to what this should be
                    n = input_to_log_abs_i.shape[1];
                    best_approx = np.zeros((n,));
                    for ii in range(n):
                        val = input_to_log_abs_i[0,ii,0,0];
                        if (val < -1):
                            best_approx[ii] = np.log1p(-val-2);
                        else:
                            best_approx[ii] = np.log1p(val);
                    sizes = 10*np.ones((n,));
                    if (True):
                        plt.figure();
                        plt.scatter(input_to_log_abs_i[0,:,0,0], log_det_jacs_i[0,:,0,0], sizes);
                        plt.scatter(input_to_log_abs_i[0,:,0,0], best_approx, sizes);
                        diff = log_det_jacs_i[0,:,0,0]-best_approx;
                        print(f_ind, np.sum(np.square(diff)));
                        plt.plot(input_to_log_abs_i[0,:,0,0], diff);
                        plt.title('layer %d' % (f_ind+1));
                        plt.show();

                """
                # check that true probability is proportional to eta*Tx
                if (exp_fam == 'normal'):
                    dist = scipy.stats.multivariate_normal(mean=mu_targ, cov=Sigma_targ);
                    log_P = dist.logpdf(_X);
                elif (exp_fam == 'dirichlet'):
                    dist = scipy.stats.dirichlet(alpha_targ);
                    log_P = dist.logpdf(_X.T);
                _Txeta = sess.run(Txeta, feed_dict);
                print(log_P.shape, _Txeta.shape);
                plt.figure();
                plt.scatter(log_P, _Txeta[:,0]);
                plt.show();
                """

                feed_dict = {Z0:z_i, eta:_eta_test};
                _X_off_lattice, _log_p_zs, _test_R2s  = sess.run([X, log_p_zs, R2s], feed_dict);
                test_R2s[check_it,:] = _test_R2s;
                # compute KL
                testing_KL = [];
                _y = np.expand_dims(_log_p_zs[0,:], 1);
                _X = _X_off_lattice[0, :, :, 0];
                testing_KL.append(approxKL(z_i[0,:,:,0], _y, _X, exp_fam, params));
                checkH(_y, exp_fam, params);
                test_KLs[check_it,:] = testing_KL;
                
                print(42*'*');
                print('it = %d ' % i);
                print('cost', cost_i);
                print('training R2', _R2s);
                print('training KL', training_KL);
                if (has_converged):
                    print('converged!!!');
                else:
                    print('not yet!!!');
                #print('saving to %s  ...' % savedir);
                #print(42*'*');
                if (np_save_flow_param):
                    if (dynamics):
                        np.savez(savedir + 'results.npz', As=As, sigma_epsilons=sigma_epsilons, flow_param_vals=flow_param_vals, cost_grad_vals=cost_grad_vals, \
                                                          flow_param_names=flow_param_names,  flow_param_nvars=flow_param_nvars, \
                                                          autocov_targ=autocov_targ, it=i, \
                                                          X=_X, X_off_lattice=_X_off_lattice, \
                                                          train_R2s=train_R2s, test_R2s=test_R2s, train_KLs=train_KLs, test_KLs=test_KLs);
                    else:
                        np.savez(savedir + 'results.npz', flow_param_vals=flow_param_vals, flow_param_names=flow_param_names,  cost_grad_vals=cost_grad_vals, \
                                                          flow_param_nvars=flow_param_nvars, it=i, \
                                                          X=_X, Tx=_Tx, log_p_zs=_log_p_zs, X_off_lattice=_X_off_lattice, \
                                                          train_R2s=train_R2s, test_R2s=test_R2s, train_KLs=train_KLs, test_KLs=test_KLs);
                else:
                    if (dynamics):
                        np.savez(savedir + 'results.npz', As=As, sigma_epsilons=sigma_epsilons, autocov_targ=autocov_targ,  \
                                                          it=i, X=_X, X_off_lattice=_X_off_lattice, \
                                                          train_R2s=train_R2s, test_R2s=test_R2s, train_KLs=train_KLs, test_KLs=test_KLs);
                    else:
                        np.savez(savedir + 'results.npz', it=i, \
                                                          X=_X, X_off_lattice=_X_off_lattice, \
                                                          train_R2s=train_R2s, test_R2s=test_R2s, train_KLs=train_KLs, test_KLs=test_KLs);
                
                check_it += 1;

                mean_train_R2 = np.mean(_R2s);
                mean_train_KL = np.mean(training_KL);
                if (has_converged):
                    print('train R2: %.3f and train KL %.3f' % (mean_train_R2, mean_train_KL));
                    break;


            sys.stdout.flush();
            i += 1;

        # save all the hyperparams
        if not os.path.exists(savedir):
                print('Making directory %s' % savedir );
                os.makedirs(savedir);
        #saveParams(params, savedir);
        # save the model
        saver.save(sess, savedir + 'model');
    if (len(_X.shape) > 2):
        assert(len(_X.shape) == 4);
        _X = _X[0, :, :, 0];
    return _log_p_zs, _X, train_R2s, train_KLs, i;
