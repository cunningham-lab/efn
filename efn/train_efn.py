# Copyright 2018 Sean Bittner, Columbia University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
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
from efn.util.efn_util import (
    get_savedir,
    construct_param_network,
    cost_fn,
    check_convergence,
    setup_param_logging,
    get_param_network_upl,
    test_convergence,
)
from tf_util.tf_util import (
    connect_density_network,
    construct_density_network,
    count_params,
    memory_extension,
)


def train_efn(
    family,
    flow_dict,
    param_net_input_type,
    K,
    M,
    stochastic_eta,
    give_hint=False,
    lr_order=-3,
    dist_seed=0,
    random_seed=0,
    min_iters=100000,
    max_iters=1000000,
    check_rate=100,
    dir_str="general",
    profile=False,
):
    """Trains an exponential family network (EFN).

        Args:
            family (obj): Instance of tf_util.families.Family.
            flow_dict (dict): Specifies structure of approximating density network.
            param_net_input_type (str): Specifies input to param network.
                'eta':        Give full eta to parameter network.
                'prior':      Part of eta that is prior-dependent.
                'likelihood': Part of eta that is likelihood-dependent.
                'data':       The data itself.
            K (int): Number of distributions per gradient descent batch.
            M (int): Number of samples per distribution per gradient descent batch.
            stochastic_eta (bool): Sample a new K distributions from the eta prior
                                   for each step of gradient descent.
            give_hint (bool): Provide hint to parameter network.
            lr_order (float): Adam learning rate is 10^(lr_order).
            dist_seed (int): Numpy random seed for drawing from eta prior.
            random_seed (int): Tensorflow random seed for initialization.
            min_iters (int): Minimum number of training interations.
            max_iters (int): Maximum number of training iterations.
            check_rate (int): Log diagonstics at every check_rate iterations.
            dir_str (str): Specifiy where to save off off '/efn/results/' filepath.
            profile (bool): Time gradient steps and save to file if True.

        """

    # Convergence criteria: mean test elbos are averaged over windows of WSIZE
    # diagnostic checks.  If the mean test elbo hasn't decreased more than
    # DELTA_THRESH, the optimization exits, provided there have already been
    # min_iters iterations.
    WSIZE = 50
    DELTA_THRESH = 1e-10

    # Diagnostic recording parameters:
    # Number of batch samples per distribution when recording model diagnostics.
    M_DIAG = int(1e3)
    # Since optimization may converge early, we dynamically allocate space to record
    # model diagnostics as optimization progresses.
    OPT_COMPRESS_FAC = 128

    # Minimum parameter network layers.
    MIN_LAYERS = 4
    # Number of parameter network layers for multivariate gaussian.
    MVN_LAYERS = 0

    # Reset tf graph, and set random seeds.
    tf.reset_default_graph()
    tf.set_random_seed(random_seed)
    np.random.seed(dist_seed)
    dist_info = {"dist_seed": dist_seed}

    # Set optimization hyperparameters.
    lr = 10 ** lr_order
    # Save tensorboard summary in intervals.
    TB_SAVE_EVERY = 50
    MODEL_SAVE_EVERY = 5000
    tb_save_params = False

    # Get the dimensionality of various components of the EFN given the parameter
    # network input type, and whether or not a hint is given to the param network.
    D_Z, num_suff_stats, num_param_net_inputs, num_T_z_inputs = family.get_efn_dims(
        param_net_input_type, give_hint
    )

    # Declare isotropic gaussian input placeholder.
    W = tf.placeholder(tf.float64, shape=(None, None, D_Z, None), name="W")
    p0 = tf.reduce_prod(
        tf.exp((-tf.square(W)) / 2.0) / np.sqrt(2.0 * np.pi), axis=[2, 3]
    )
    base_log_q_z = tf.log(p0[:, :])

    # Assemble density network.
    flow_layers, num_theta_params = construct_density_network(flow_dict, D_Z, family.T)
    flow_layers, num_theta_params = family.map_to_support(flow_layers, num_theta_params)

    # Set number of layers in the parameter network.
    if family.name == "normal":
        L = MVN_LAYERS
    else:
        # We use square root scaling of layer count with density network dimensionality,
        # while enforcing a minimum of MIN_LAYERS layers.
        L = max(int(np.ceil(np.sqrt(D_Z))), MIN_LAYERS)
    # The number of units per layer is a linear interpolation between the dimensionality
    # of the parameter network input and the number of parameters in the density network.
    upl_tau = None
    upl_shape = "linear"
    upl = get_param_network_upl(
        L, num_param_net_inputs, num_theta_params, upl_tau, upl_shape
    )
    param_net_hps = {"L": L, "upl": upl}

    # Declare EFN input placeholders.
    eta = tf.placeholder(tf.float64, shape=(None, num_suff_stats))
    param_net_input = tf.placeholder(tf.float64, shape=(None, num_param_net_inputs))
    T_z_input = tf.placeholder(tf.float64, shape=(None, num_T_z_inputs))

    if not stochastic_eta:
        # Use the same fixed training and testing eta lattice.
        _eta, _param_net_input, _T_z_input, eta_draw_params = family.draw_etas(
            K, param_net_input_type, give_hint
        )
        _eta_test = _eta
        _param_net_input_test = _param_net_input
        _T_z_input_test = _T_z_input
        eta_test_draw_params = eta_draw_params
    else:
        # Draw the eta test lattice, which will remain fixed throughout training.
        np.random.seed(0)
        if family.name == "log_gaussian_cox":
            _eta_test, _param_net_input_test, _T_z_input_test, eta_test_draw_params = family.draw_etas(
                K, param_net_input_type, give_hint, train=False
            )
        else:
            _eta_test, _param_net_input_test, _T_z_input_test, eta_test_draw_params = family.draw_etas(
                K, param_net_input_type, give_hint
            )

    # Create model save directory if doesn't exist.
    efn_str = "EFN1" if (K == 1) else "EFN"
    savedir = get_savedir(
        family,
        efn_str,
        dir_str,
        param_net_input_type,
        K,
        M,
        flow_dict,
        param_net_hps,
        give_hint,
        random_seed,
        dist_info,
    )
    if not os.path.exists(savedir):
        print("Making directory %s" % savedir)
        os.makedirs(savedir)

    # Construct the parameter network.
    theta = construct_param_network(param_net_input, K, flow_layers, param_net_hps)

    # Connect parameter network to the density network.
    Z, sum_log_det_jacobian, Z_by_layer = connect_density_network(W, flow_layers, theta)
    log_q_zs = base_log_q_z - sum_log_det_jacobian

    all_params = tf.trainable_variables()
    nparams = len(all_params)

    # Compute family-specific sufficient statistics and log base measure on samples.
    T_z = family.compute_suff_stats(Z, Z_by_layer, T_z_input)
    log_h_z = family.compute_log_base_measure(Z)

    # Compute total cost and ELBOS and r^2 for each distribution.
    cost, elbos, R2s = cost_fn(eta, log_q_zs, T_z, log_h_z, K)

    # Compute gradient of parameter network params (phi) wrt cost.
    # Cost is KL_p(eta)(q(z; eta) || p[(z; eta)).
    cost_grad = tf.gradients(cost, all_params)
    grads_and_vars = []
    for i in range(len(all_params)):
        grads_and_vars.append((cost_grad[i], all_params[i]))

    # Add inputs and outputs of EFN to saved tf model.
    tf.add_to_collection("W", W)
    tf.add_to_collection("Z", Z)
    tf.add_to_collection("eta", eta)
    tf.add_to_collection("param_net_input", param_net_input)
    tf.add_to_collection("log_q_zs", log_q_zs)
    tf.add_to_collection("T_z_input", T_z_input)
    saver = tf.train.Saver()

    # Tensorboard logging.
    summary_writer = tf.summary.FileWriter(savedir)
    tf.summary.scalar("cost", cost)
    if tb_save_params:
        setup_param_logging(all_params)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_step = optimizer.apply_gradients(grads_and_vars)

    summary_op = tf.summary.merge_all()

    # Key diagnostics recorded throughout optimization.  train_* records the diagnostic
    # for the K samples from the eta prior from the most recent batch sample.  test_*
    # records model diagnostics on K-distributions of a fixed testing set drawn from the
    # eta prior.
    max_diagnostic_checks = (max_iters // check_rate) + 1
    array_init_len = int(np.ceil(max_diagnostic_checks / OPT_COMPRESS_FAC))
    array_cur_len = array_init_len
    train_elbos = np.zeros((array_init_len, K))
    test_elbos = np.zeros((array_init_len, K))
    train_R2s = np.zeros((array_init_len, K))
    test_R2s = np.zeros((array_init_len, K))
    train_KLs = np.zeros((array_init_len, K))
    test_KLs = np.zeros((array_init_len, K))

    # If profiling the code, keep track of the time it takes to compute each gradient.
    if profile:
        grad_comp_times = np.zeros((max_iters,))

    check_it = 0
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # compute R^2, KL, and elbo for training set
        w_i = np.random.normal(np.zeros((K, M, D_Z, family.T)), 1.0)
        if stochastic_eta:
            _eta, _param_net_input, _T_z_input, eta_draw_params = family.draw_etas(
                K, param_net_input_type, give_hint
            )
        feed_dict_train = {
            W: w_i,
            eta: _eta,
            param_net_input: _param_net_input,
            T_z_input: _T_z_input,
        }
        summary = sess.run(summary_op, feed_dict_train)
        summary_writer.add_summary(summary, 0)
        train_elbos[check_it, :], train_R2s[check_it, :], train_KLs[
            check_it, :
        ], train_Z = family.batch_diagnostics(
            K, sess, feed_dict_train, Z, log_q_zs, elbos, R2s, eta_draw_params
        )

        # compute R^2, KL, and elbo for static test set
        feed_dict_test = {
            W: w_i,
            eta: _eta_test,
            param_net_input: _param_net_input_test,
            T_z_input: _T_z_input_test,
        }
        test_elbos[check_it, :], test_R2s[check_it, :], test_KLs[
            check_it, :
        ], test_Z = family.batch_diagnostics(
            K, sess, feed_dict_test, Z, log_q_zs, elbos, R2s, eta_test_draw_params
        )

        check_it += 1
        i = 1

        print("Starting EFN optimization.")
        while i < max_iters:
            # Draw a new noise tensor.
            w_i = np.random.normal(np.zeros((K, M, D_Z, family.T)), 1.0)
            # Draw a new set of K etas from the eta prior (unless fixed).
            if stochastic_eta:
                _eta, _param_net_input, _T_z_input, eta_draw_params = family.draw_etas(
                    K, param_net_input_type, give_hint
                )
            feed_dict = {
                W: w_i,
                eta: _eta,
                param_net_input: _param_net_input,
                T_z_input: _T_z_input,
            }

            if profile:
                # Time a single gradient step.
                start_time = time.time()
                ts = sess.run(train_step, feed_dict)
                end_time = time.time()
                seconds = end_time - start_time
                print("iter %d took %f seconds" % (i + 1, seconds))
                times[i] = seconds
                i += 1
                continue
            else:
                # Take a gradient step.
                if np.mod(i, check_rate) == 0:
                    start_time = time.time()
                ts, cost_i, summary = sess.run(
                    [train_step, cost, summary_op], feed_dict
                )
                if np.mod(i, check_rate) == 0:
                    end_time = time.time()
                    print("iter %d took %f seconds" % (i + 1, end_time - start_time))

            # Write to tensorboard periodically.
            if np.mod(i, TB_SAVE_EVERY) == 0:
                summary_writer.add_summary(summary, i)

            # Save model periodically.
            if np.mod(i, MODEL_SAVE_EVERY) == 0:
                saver.save(sess, savedir + "model")

            # Log and print diagnostic information periodically.
            if np.mod(i + 1, check_rate) == 0:
                print(42 * "*")
                print(savedir)
                print("it = %d " % (i + 1))
                start_time = time.time()

                # compute R^2, KL, and elbo for training set
                w_i = np.random.normal(np.zeros((K, M_DIAG, D_Z, family.T)), 1.0)
                feed_dict_train = {
                    W: w_i,
                    eta: _eta,
                    param_net_input: _param_net_input,
                    T_z_input: _T_z_input,
                }
                train_elbos[check_it, :], train_R2s[check_it, :], train_KLs[
                    check_it, :
                ], train_Z = family.batch_diagnostics(
                    K, sess, feed_dict_train, Z, log_q_zs, elbos, R2s, eta_draw_params
                )

                # compute R^2, KL, and elbo for static test set
                feed_dict_test = {
                    W: w_i,
                    eta: _eta_test,
                    param_net_input: _param_net_input_test,
                    T_z_input: _T_z_input_test,
                }
                test_elbos[check_it, :], test_R2s[check_it, :], test_KLs[
                    check_it, :
                ], test_Z = family.batch_diagnostics(
                    K,
                    sess,
                    feed_dict_test,
                    Z,
                    log_q_zs,
                    elbos,
                    R2s,
                    eta_test_draw_params,
                )

                print("train elbo: %f" % np.mean(train_elbos[check_it, :]))
                print("train R2: %f" % np.mean(train_R2s[check_it, :]))
                if family.name in ["dirichlet", "normal", "inv_wishart"]:
                    print("train KL: %f" % np.mean(train_KLs[check_it, :]))

                print("test elbo: %f" % np.mean(test_elbos[check_it, :]))
                print("test R2: %f" % np.mean(test_R2s[check_it, :]))
                if family.name in ["dirichlet", "normal", "inv_wishart"]:
                    print("test KL: %f" % np.mean(test_KLs[check_it, :]))

                np.savez(
                    savedir + "results.npz",
                    it=i,
                    check_rate=check_rate,
                    train_Z=train_Z,
                    test_Z=test_Z,
                    eta=_eta,
                    eta_dist=family.eta_dist,
                    param_net_input=_param_net_input,
                    train_params=eta_draw_params,
                    test_params=eta_test_draw_params,
                    T_z_input=_T_z_input,
                    converged=False,
                    train_elbos=train_elbos,
                    test_elbos=test_elbos,
                    train_R2s=train_R2s,
                    test_R2s=test_R2s,
                    train_KLs=train_KLs,
                    test_KLs=test_KLs,
                    final_cost=cost_i,
                )
                if family.name == "lgc":
                    np.savez(
                        savedir + "data_info.npz",
                        test_set=family.test_set,
                        train_set=family.train_set,
                    )

                # Test for convergence.
                if check_it >= 2 * WSIZE - 1:
                    mean_test_elbos = np.mean(test_elbos, 1)
                    if i >= min_iters:
                        if test_convergence(
                            mean_test_elbos, check_it, WSIZE, DELTA_THRESH
                        ):
                            print("converged!")
                            break

                check_it += 1
                # Dynamically extend memory if necessary.
                if check_it == array_cur_len:
                    print(
                        "Extending log length from %d to %d"
                        % (array_cur_len, 2 * array_cur_len)
                    )
                    print(
                        "Extending log length from %d to %d"
                        % (array_cur_len, 2 * array_cur_len)
                    )
                    print(
                        "Extending log length from %d to %d"
                        % (array_cur_len, 2 * array_cur_len)
                    )
                    print(
                        "Extending log length from %d to %d"
                        % (array_cur_len, 2 * array_cur_len)
                    )
                    train_elbos, train_R2s, train_KLs, test_elbos, test_R2s, test_KLs = memory_extension(
                        [
                            train_elbos,
                            train_R2s,
                            train_KLs,
                            test_elbos,
                            test_R2s,
                            test_KLs,
                        ],
                        array_cur_len,
                    )
                    array_cur_len = 2 * array_cur_len
            sys.stdout.flush()
            i += 1

        # Save profiling information if profiling.
        if profile:
            pfname = savedir + "profile.npz"
            np.savez(pfname, times=times)
            exit()

        # Save model.
        print("Saving model before exitting.")
        saver.save(sess, savedir + "model")

    # Save training diagnostics and model info.
    if i < max_iters:
        np.savez(
            savedir + "results.npz",
            it=i,
            check_rate=check_rate,
            train_Z=train_Z,
            test_Z=test_Z,
            eta=_eta,
            eta_dist=family.eta_dist,
            param_net_input=_param_net_input,
            train_params=eta_draw_params,
            test_params=eta_test_draw_params,
            T_z_input=_T_z_input,
            converged=True,
            train_elbos=train_elbos,
            test_elbos=test_elbos,
            train_R2s=train_R2s,
            test_R2s=test_R2s,
            train_KLs=train_KLs,
            test_KLs=test_KLs,
            final_cost=cost_i,
        )
        if family.name == "lgc":
            np.savez(
                savedir + "data_info.npz",
                test_set=family.test_set,
                train_set=family.train_set,
            )

    return None
