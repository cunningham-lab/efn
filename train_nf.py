import tensorflow as tf
import numpy as np
import time
import csv
import scipy.stats
import sys
import os
import io
from sklearn.metrics import pairwise_distances
from statsmodels.tsa.ar_model import AR
from efn_util import setup_IO, log_grads, cost_fn, test_convergence, \
                     memory_extension, setup_param_logging
from tf_util.tf_util import connect_density_network, construct_density_network, declare_theta, count_params

def train_nf(family, params, flow_dict, M=1000, lr_order=-3, random_seed=0, \
             min_iters=100000, max_iters=1000000, check_rate=100, dir_str='general', profile=False):
    """Trains an normalizing flow (NF).

        Args:
            family (obj): Instance of tf_util.families.Family.
            params (dict): Mean parameters of distribution to learn with NF.
            flow_dict (dict): Specifies structure of approximating density network.
            M (int): Number of samples per distribution per gradient descent batch.
            lr_order (float): Adam learning rate is 10^(lr_order).
            random_seed (int): Tensorflow random seed for initialization.
            min_iters (int): Minimum number of training interations.
            max_iters (int): Maximum number of training iterations.
            check_rate (int): Log diagonstics at every check_rate iterations.
            dir_str (str): Specifiy where to save off off '/efn/results/' filepath.
            profile (bool): Time gradient steps and save to file if True.

        """

    # Learn a single (K=1) distribution with an NF.
    K = 1;

    # Convergence criteria: training elbos are averaged over windows of wsize 
    # diagnostic checks.  If the elbo hasn't decreased more than delta_thresh, 
    # the optimization exits, provided there have already been min_iters iterations.
    wsize = 50;
    delta_thresh = 1e-10;

    # Reset tf graph, and set random seeds.
    tf.reset_default_graph();
    tf.set_random_seed(random_seed);
    np.random.seed(0);
    dist_info = {'dist_seed':params['dist_seed']};

    # Set optimization hyperparameters.
    lr = 10**lr_order
    # Save tensorboard summary in intervals.
    tb_save_every = 50;
    model_save_every = 5000;
    tb_save_params = False;

    # Get the dimensionality of various components of the EFN given the parameter
    # network input type, and whether or not a hint is given to the param network.
    D_Z, num_suff_stats, num_param_net_inputs, num_T_z_inputs = family.get_efn_dims('eta', False);

    # Declare isotropic gaussian input placeholder.
    W = tf.placeholder(tf.float64, shape=(None, None, D_Z, None), name='W');
    p0 = tf.reduce_prod(tf.exp((-tf.square(W))/2.0)/np.sqrt(2.0*np.pi), axis=[2,3]); 
    base_log_q_z = tf.log(p0[:,:]);

    # Assemble density network.
    flow_layers, num_theta_params = construct_density_network(flow_dict, D_Z, family.T);
    flow_layers, num_theta_params = family.map_to_support(flow_layers, num_theta_params);

    # Declare NF input placeholders.
    eta = tf.placeholder(tf.float64, shape=(None, num_suff_stats), name='eta');
    T_z_input = tf.placeholder(tf.float64, shape=(None, num_T_z_inputs), name='T_z_input');

    # Create model save directory if doesn't exist.
    savedir = setup_IO(family, 'NF1', dir_str, '', K, M, flow_dict, {}, False, random_seed, dist_info);
    if not os.path.exists(savedir):
        print('Making directory %s' % savedir );
        os.makedirs(savedir);

    # Get eta and T(z) given the distribution mean parameters.
    _eta, _ = family.mu_to_eta(params, 'eta', False);
    _eta = np.expand_dims(_eta, 0);
    _T_z_input = family.mu_to_T_z_input(params);
    _T_z_input = np.expand_dims(_T_z_input, 0);
    # Test distribution is the same as training
    _eta_test = _eta;
    _T_z_input_test = _T_z_input;

    # Declare density network parameters.
    theta = declare_theta(flow_layers);

    # connect time-invariant flow
    Z, sum_log_det_jacobian, Z_by_layer = connect_density_network(W, flow_layers, theta);
    log_q_zs = base_log_q_z - sum_log_det_jacobian;

    # generative model is fully specified
    all_params = tf.trainable_variables();
    nparams = len(all_params);

    # set up the constraint computation
    T_z = family.compute_suff_stats(Z, Z_by_layer, T_z_input);
    log_h_z = family.compute_log_base_measure(Z);
    # exponential family optimization
    cost, costs, R2s = cost_fn(eta, log_q_zs, T_z, log_h_z, K)
    cost_grad = tf.gradients(cost, all_params);

    grads_and_vars = [];
    for i in range(len(all_params)):
        grads_and_vars.append((cost_grad[i], all_params[i]));

    # set optimization hyperparameters
    tf.add_to_collection('W', W);
    tf.add_to_collection('Z', Z);
    tf.add_to_collection('eta', eta);
    tf.add_to_collection('log_q_zs', log_q_zs);
    tf.add_to_collection('T_z_input', T_z_input);
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

    if (profile):
        times = np.zeros((max_iters,));

    check_it = 0;
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer();
        sess.run(init_op);

        w_i = np.random.normal(np.zeros((K, M, D_Z, family.T)), 1.0);
        feed_dict = {W:w_i, eta:_eta, T_z_input:_T_z_input};

        cost_i, summary = \
            sess.run([cost, summary_op], feed_dict);

        w_i = np.random.normal(np.zeros((K, int(1e3), D_Z, family.T)), 1.0);
        feed_dict_train = {W:w_i, eta:_eta, T_z_input:_T_z_input};
        train_elbos[check_it,:], train_R2s[check_it,:], train_KLs[check_it,:], train_Z = family.batch_diagnostics(K, sess, feed_dict_train, Z, log_q_zs, costs, R2s, [params], True);
        feed_dict_test = {W:w_i, eta:_eta_test, T_z_input:_T_z_input_test};
        test_elbos[check_it,:], test_R2s[check_it,:], test_KLs[check_it,:], test_Z = family.batch_diagnostics(K, sess, feed_dict_test, Z, log_q_zs, costs, R2s, [params]);

        check_it += 1;

        optimizer = tf.train.AdamOptimizer(learning_rate=lr);

        train_step = optimizer.apply_gradients(grads_and_vars);
        
        init_op = tf.global_variables_initializer();
        sess.run(init_op);

        i = 1;

        has_converged = False;
        while (i < max_iters):
            w_i = np.random.normal(np.zeros((K, M, D_Z, family.T)), 1.0);
            feed_dict = {W:w_i, eta:_eta, T_z_input:_T_z_input};

            if (profile):
                start_time = time.time();
                ts = sess.run(train_step, feed_dict);
                end_time = time.time();
                seconds = end_time-start_time;
                print('iter %d took %f seconds' % (i+1, seconds));
                times[i] = seconds;
                i += 1;
                continue;
            else:
                if (np.mod(i, check_rate)==0):
                    start_time = time.time();
                ts, cost_i, _cost_grads, _R2s, _T_z, _T_z, summary = \
                        sess.run([train_step, cost, cost_grad, R2s, T_z, log_h_z, summary_op], feed_dict);
                if (np.mod(i, check_rate)==0):
                    end_time = time.time();
                    print('iter %d took %f seconds' % (i+1, end_time-start_time));


            if (np.mod(i,tb_save_every)==0):
                print('saving summary', i);
                summary_writer.add_summary(summary, i);

            if (np.mod(i,model_save_every) == 0):
                # save all the hyperparams
                if not os.path.exists(savedir):
                    print('Making directory %s' % savedir );
                    os.makedirs(savedir);
                print('saving model at iter', i);
                saver.save(sess, savedir + 'model');

            if (np.mod(i+1, check_rate)==0):
                w_i = np.random.normal(np.zeros((K, int(1e3), D_Z, family.T)), 1.0);
                feed_dict_train = {W:w_i, eta:_eta, T_z_input:_T_z_input};
                train_elbos[check_it,:], train_R2s[check_it,:], train_KLs[check_it,:], train_Z = family.batch_diagnostics(K, sess, feed_dict_train, Z, log_q_zs, costs, R2s, [params], True);
                feed_dict_test = {W:w_i, eta:_eta_test, T_z_input:_T_z_input_test};
                test_elbos[check_it,:], test_R2s[check_it,:], test_KLs[check_it,:], test_Z = family.batch_diagnostics(K, sess, feed_dict_test, Z, log_q_zs, costs, R2s, [params]);

                print(42*'*');
                print(savedir);
                print('it = %d ' % (i+1));
                print('cost = %f ' % cost_i);
                print('train elbo %.3f, train R2: %.3f, train KL %.3f' % (np.mean(train_elbos[check_it,:]), np.mean(train_R2s[check_it,:]), np.mean(train_KLs[check_it,:])));
                
                np.savez(savedir + 'results.npz', it=i, Z=train_Z, eta=_eta, \
                                                  T_z_input=_T_z_input, params=params, check_rate=check_rate, \
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

        w_i = np.random.normal(np.zeros((K, M, D_Z, family.T)), 1.0);
        feed_dict = {W:w_i, eta:_eta, T_z_input:_T_z_input};
        _log_q_zs, _X = sess.run([log_q_zs, X], feed_dict);

        if (profile):
            pfname = savedir + 'profile.npz';
            np.savez(pfname, times=times, M=M);
            exit();

    if (i < max_iters):
        np.savez(savedir + 'results.npz', it=i, X=train_X, eta=_eta, T_z_input=_T_z_input, params=params, check_rate=check_rate, \
                                      train_elbos=train_elbos, test_elbos=test_elbos, M=M, \
                                      train_R2s=train_R2s, test_R2s=test_R2s, \
                                      train_KLs=train_KLs, test_KLs=test_KLs, \
                                      converged=True, final_cos=cost_i);

    return None;
