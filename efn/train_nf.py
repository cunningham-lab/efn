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
    cost_fn,
    test_convergence,
    setup_param_logging,
)
from tf_util.tf_util import (
    density_network,
    memory_extension,
)


def train_nf(
    family,
    params,
    arch_dict,
    M=1000,
    lr_order=-3,
    random_seed=0,
    min_iters=100000,
    max_iters=1000000,
    check_rate=100,
    dir_str="general",
    profile=False,
    savedir=None,
):
    """Trains an normalizing flow (NF).

        Args:
            family (obj): Instance of tf_util.families.Family.
            params (dict): Mean parameters of distribution to learn with NF.
            arch_dict (dict): Specifies structure of approximating density network.
            M (int): Number of samples per distribution per gradient descent batch.
            lr_order (float): Adam learning rate is 10^(lr_order).
            random_seed (int): Tensorflow random seed for initialization.
            min_iters (int): Minimum number of training interations.
            max_iters (int): Maximum number of training iterations.
            check_rate (int): Log diagonstics at every check_rate iterations.
            dir_str (str): Specifiy where to save off off '/efn/models/' filepath.
            profile (bool): Time gradient steps and save to file if True.
            savedir (str): If not None, use this as model save directory.

        """

    # Learn a single (K=1) distribution with an NF.
    K = 1
    D_Z = family.D_Z

    # Convergence criteria: training elbos are averaged over windows of WSIZE
    # diagnostic checks.  If the elbo hasn't decreased more than DELTA_THRESH,
    # the optimization exits, provided there have already been min_iters iterations.
    WSIZE = 50
    DELTA_THRESH = 1e-10

    # Diagnostic recording parameters:
    # Number of batch samples per distribution when recording model diagnostics.
    M_DIAG = int(1e3)
    # Since optimization may converge early, we dynamically allocate space to record
    # model diagnostics as optimization progresses.
    OPT_COMPRESS_FAC = 16

    # Save fisher information matrix
    FIM = True

    # Reset tf graph, and set random seeds.
    tf.reset_default_graph()
    tf.set_random_seed(random_seed)
    np.random.seed(0)
    dist_info = {"dist_seed": params["dist_seed"]}

    # Set optimization hyperparameters.
    lr = 10 ** lr_order
    # Save tensorboard summary in intervals.
    tb_save_every = 50
    model_save_every = 5000
    tb_save_params = False

    # Get the dimensionality of various components of the EFN given the parameter 
    # network input type, and whether or not a hint is given to the param network.  
    D_Z, num_suff_stats, num_param_net_inputs, num_T_z_inputs = family.get_efn_dims( "eta", False)

    # Declare isotropic gaussian input placeholder.
    W = tf.placeholder(tf.float64, shape=(None, None, D_Z), name="W")
    p0 = tf.reduce_prod(
        tf.exp((-tf.square(W)) / 2.0) / np.sqrt(2.0 * np.pi), axis=2
    )
    base_log_q_z = tf.log(p0)

    # Declare NF input placeholders.
    eta = tf.placeholder(tf.float64, shape=(None, num_suff_stats), name="eta")
    T_z_input = tf.placeholder(
        tf.float64, shape=(None, num_T_z_inputs), name="T_z_input"
    )

    # Create model save directory if doesn't exist.
    if (savedir is None):
        savedir = get_savedir(
            family, "NF1", dir_str, "", K, M, arch_dict, {}, False, random_seed, dist_info
        )
    if not os.path.exists(savedir):
        print("Making directory %s" % savedir)
        os.makedirs(savedir)

    # Get eta and T(z) given the distribution mean parameters.
    _eta, _ = family.mu_to_eta(params, "eta", False)
    _eta = np.expand_dims(_eta, 0)
    _T_z_input = family.mu_to_T_z_input(params)
    _T_z_input = np.expand_dims(_T_z_input, 0)

    if (family.has_support_map):
        support_mapping = family.support_mapping
    else:
        support_mapping = None
    # Construct density network parameters.
    Z, sum_log_det_jacobian, flow_layers = density_network(W, arch_dict, support_mapping)
    log_q_zs = base_log_q_z - sum_log_det_jacobian

    # Permutations and batch norms
    final_thetas = {};
    batch_norm_mus = []
    batch_norm_sigmas = []
    batch_norm_layer_means = []
    batch_norm_layer_vars = []
    _batch_norm_mus = []
    _batch_norm_sigmas = []
    batch_norm = False
    for i in range(len(flow_layers)):
        flow_layer = flow_layers[i]
        if (flow_layer.name == 'PermutationFlow'):
            final_thetas.update({'DensityNetwork/Layer%d/perm_inds' % (i+1):flow_layer.inds})
        if (flow_layer.name == 'RealNVP' and flow_layer.batch_norm):
            batch_norm = True
            num_masks = arch_dict['real_nvp_arch']['num_masks']
            for j in range(num_masks):
                batch_norm_mus.append(flow_layer.mus[j])
                batch_norm_sigmas.append(flow_layer.sigmas[j])
                batch_norm_layer_means.append(flow_layer.layer_means[j])
                batch_norm_layer_vars.append(flow_layer.layer_vars[j])
                _batch_norm_mus.append(np.zeros((D_Z,)))
                _batch_norm_sigmas.append(np.ones((D_Z,)))
                
    print('Saved', final_thetas.keys())

    
    if (FIM):
        if (arch_dict['flow_type'] == 'RealNVP'):
            print('computing inverse of realNVP')
            Z_INV = Z
            layer_ind = len(flow_layers) - 1
            if (family.has_support_map):
                support_mapping = flow_layers[layer_ind]
                Z_INV = support_mapping.inverse(Z_INV)
                layer_ind = layer_ind - 1

            if (arch_dict['post_affine']):
                # unshift
                shift_layer = flow_layers[layer_ind]
                Z_INV = (Z_INV - tf.expand_dims(shift_layer.b, 1))
                layer_ind = layer_ind - 1

                # unmult
                elem_mult_layer = flow_layers[layer_ind]
                Z_INV = Z_INV / tf.expand_dims(elem_mult_layer.a, 1)
                layer_ind = layer_ind - 1

            while (layer_ind > -1):
                layer = flow_layers[layer_ind]
                Z_INV = layer.inverse(Z_INV)
                layer_ind -= 1

        else:
            Z_INV = tf.placeholder(tf.float64, (1,))

    all_params = tf.trainable_variables()
    nparams = len(all_params)

    Z_by_layer = []
    # Compute family-specific sufficient statistics and log base measure on samples.
    T_z = family.compute_suff_stats(Z, Z_by_layer, T_z_input)
    if (family.constant_base_measure):
        log_h_z = 0.0
    else:
        log_h_z = family.compute_log_base_measure(Z)

    # Compute total cost, and ELBO and r^2 for K=1 distribution.
    cost, elbos, R2s = cost_fn(eta, log_q_zs, T_z, log_h_z, K)

    # Compute gradient of density network params (theta) wrt cost.
    # Cost is KL_eta(q(z; eta) || p[(z; eta)).
    cost_grad = tf.gradients(cost, all_params)
    grads_and_vars = []
    for i in range(len(all_params)):
        grads_and_vars.append((cost_grad[i], all_params[i]))

    # Add inputs and outputs of NF to saved tf model.
    tf.add_to_collection("W", W)
    tf.add_to_collection("Z", Z)
    if (FIM):
        tf.add_to_collection("Z_INV", Z_INV)
    tf.add_to_collection("eta", eta)
    tf.add_to_collection("log_q_zs", log_q_zs)
    tf.add_to_collection("T_z_input", T_z_input)

    saver = tf.train.Saver(all_params)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_step = optimizer.apply_gradients(grads_and_vars)

    # Tensorboard logging.
    summary_writer = tf.summary.FileWriter(savedir)
    tf.summary.scalar("cost", cost)
    if tb_save_params:
        setup_param_logging(all_params)

    summary_op = tf.summary.merge_all()

    # Key diagnostics recorded throughout optimization.  train_* records the diagnostic
    # for K = 1 distribution we are optimizing.
    max_diagnostic_checks = (max_iters // check_rate) + 1
    array_init_len = int(np.ceil(max_diagnostic_checks / OPT_COMPRESS_FAC))
    print("array_init_len", array_init_len)
    array_cur_len = array_init_len
    train_elbos = np.zeros((array_init_len, K))
    train_R2s = np.zeros((array_init_len, K))
    train_KLs = np.zeros((array_init_len, K))

    # If profiling the code, keep track of the time it takes to compute each gradient.
    if profile:
        times = np.zeros((max_iters,))

    check_it = 0
    converged = False
    print('beginning session')
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        print('in session')
        sess.run(init_op)
        print('base initialization')

        # compute R^2, KL, and elbo
        w_i = np.random.normal(np.zeros((K, M, D_Z)), 1.0)
        feed_dict = {W: w_i, eta: _eta, T_z_input: _T_z_input}

        # Initialize the batch norms.  Iteratively run out the coupling layers.
        if (batch_norm):
            print('Initializing batch norm parameters.')
            num_batch_norms = len(batch_norm_mus)
            for j in range(num_batch_norms):
                _batch_norm_mus[j] = sess.run(batch_norm_layer_means[j], feed_dict)
                _batch_norm_sigmas[j] = np.sqrt(sess.run(batch_norm_layer_vars[j], feed_dict))
                feed_dict.update({batch_norm_mus[j]:_batch_norm_mus[j]})
                feed_dict.update({batch_norm_sigmas[j]:_batch_norm_sigmas[j]})

        summary = sess.run(summary_op, feed_dict)
        summary_writer.add_summary(summary, 0)
        train_elbos[check_it, :], train_R2s[check_it, :], train_KLs[
            check_it, :
        ], train_Z = family.batch_diagnostics(
            K, sess, feed_dict, Z, log_q_zs, elbos, R2s, [params], True
        )
        print('after bd')

        check_it += 1
        i = 1

        print("Starting NF optimization.")
        while i < max_iters:
            # Draw a new noise tensor.
            w_i = np.random.normal(np.zeros((K, M, D_Z)), 1.0)
            feed_dict.update({W: w_i})

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

                args = [train_step, cost, cost_grad, R2s, T_z, summary_op]
                if (batch_norm):
                    args += [batch_norm_layer_means, batch_norm_layer_vars]
                _args = sess.run(args, feed_dict)
                cost_i = _args[1]
                _cost_grads = _args[2]
                _R2s = _args[3]
                _T_z = _args[4]
                summary = _args[5]

                # Update batch norm params
                if (batch_norm):
                    mom = 0.99

                    _batch_norm_layer_means = _args[6]
                    _batch_norm_layer_vars = _args[7]
                    for j in range(num_batch_norms):
                        _batch_norm_mus[j] = mom*_batch_norm_mus[j] + (1.0-mom)*_batch_norm_layer_means[j]
                        _batch_norm_sigmas[j] = mom*_batch_norm_sigmas[j] + (1.0-mom)*np.sqrt(_batch_norm_layer_vars[j])

                if np.mod(i, check_rate) == 0:
                    end_time = time.time()
                    print("iter %d took %f seconds" % (i + 1, end_time - start_time))

            # Write to tensorboard periodically.
            if np.mod(i, tb_save_every) == 0:
                summary_writer.add_summary(summary, i)

            # Save model periodically.
            if np.mod(i, model_save_every) == 0:
                print("Saving model at iter %d." % i)
                saver.save(sess, savedir + "model")

            # Log and print diagnostic information periodically.
            if np.mod(i + 1, check_rate) == 0:
                print(42 * "*")
                print(savedir)
                print("it = %d " % (i + 1))
                w_i = np.random.normal(np.zeros((K, M_DIAG, D_Z)), 1.0)
                feed_dict.update({W: w_i})
                train_elbos[check_it, :], train_R2s[check_it, :], train_KLs[
                    check_it, :
                ], train_Z = family.batch_diagnostics(
                    K, sess, feed_dict, Z, log_q_zs, elbos, R2s, [params], True
                )

                print("train elbo: %f" % np.mean(train_elbos[check_it, :]))
                print("train R2: %f" % np.mean(train_R2s[check_it, :]))
                if family.name in ["dirichlet", "normal", "inv_wishart"]:
                    print("train KL: %f" % np.mean(train_KLs[check_it, :]))

                np.savez(
                    savedir + "opt_info.npz",
                    it=i,
                    Z=train_Z,
                    eta=_eta,
                    T_z_input=_T_z_input,
                    params=params,
                    check_rate=check_rate,
                    train_elbos=train_elbos,
                    train_R2s=train_R2s,
                    train_KLs=train_KLs,
                    converged=converged,
                    final_cos=cost_i,
                )

                # Test for convergence.
                if check_it >= 2 * WSIZE - 1:
                    mean_train_elbos = np.mean(train_elbos, 1)
                    if i >= min_iters:
                        if test_convergence(
                            mean_train_elbos, check_it, WSIZE, DELTA_THRESH
                        ):
                            print("converged!")
                            converged = True
                            break

                check_it += 1

                # Dynamically extend memory if necessary.
                if check_it == array_cur_len:
                    print(
                        "Extending log length from %d to %d"
                        % (array_cur_len, 2 * array_cur_len)
                    )
                    train_elbos, train_R2s, train_KLs = memory_extension(
                        [train_elbos, train_R2s, train_KLs], array_cur_len
                    )
                    array_cur_len = 2 * array_cur_len

            if (batch_norm):
                for j in range(num_batch_norms):
                    feed_dict.update({batch_norm_mus[j]:_batch_norm_mus[j]})
                    feed_dict.update({batch_norm_sigmas[j]:_batch_norm_sigmas[j]})

            sys.stdout.flush()
            i += 1

        # Save profiling information if profiling.
        if profile:
            pfname = savedir + "profile.npz"
            np.savez(pfname, times=times, M=M)
            exit()

        # Save model.
        print("Saving model before exitting.")
        print("Saving model at iter %d." % i)
        saver.save(sess, savedir + "model")

        # save parameters
        for i in range(nparams):
            final_thetas.update({all_params[i].name:sess.run(all_params[i])});
            for j in range(num_batch_norms):
                final_thetas.update({'DensityNetwork/batch_norm_mu%d' % (j+1):_batch_norm_mus[j]})
                final_thetas.update({'DensityNetwork/batch_norm_sigma%d' % (j+1):_batch_norm_sigmas[j]})

        np.savez(
                savedir + "theta.npz",
                theta=final_thetas
            )

    # Save training diagnostics and model info.
    if i < max_iters:
        np.savez(
            savedir + "opt_info.npz",
            it=i,
            M=M,
            Z=train_Z,
            eta=_eta,
            T_z_input=_T_z_input,
            params=params,
            check_rate=check_rate,
            train_elbos=train_elbos,
            train_R2s=train_R2s,
            train_KLs=train_KLs,
            converged=converged,
            final_cost=cost_i,
        )

    return converged
