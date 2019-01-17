import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.tri as tri
from scipy.stats import multivariate_normal
from functools import reduce
from tf_util.families import family_from_str
import os
import pandas as pd
from itertools import compress
import scipy.io as sio

#from dirichlet import simplex
colors = ["pale red", "medium green", "windows blue", "amber", "dusty purple", "greyish", "faded green", "denim blue"];


def report_model_status(fname, elbo_nz, R2_nz, elbo_nans, R2_nans, file_exists):
    if (not file_exists):
        return 1;
    elif (elbo_nz > 0 or R2_nz > 0):
        return 2;
    elif (elbo_nans > 0 or R2_nans > 0):
        return 3;
    return 4

def find_last_ind(x):
    max_checks = x.shape[0];
    K = x.shape[1];
    found = False;
    dec = -1;
    while (not found and (dec < max_checks)):
        dec = dec+1;
        vals = x[max_checks-dec-1,:];
        vals_nz = np.sum(vals == 0);
        found = not (vals_nz == K);
    return max_checks-dec-1;
    
def get_seed(exp_fam, D, ds):
    if (exp_fam == 'normal'):
        seeds = range(1,11);
    elif (exp_fam == 'dirichlet'):
        seeds = [1,2,3,4,5,6,6,7,8,9,11];
    else:
        seeds = range(10);
    return seeds[ds];

    
def get_latest_diagnostics(fname, is_train=True, ind=None):
    wsize = 50;
    delta_thresh = 1e-10;
    file_exists = os.path.isfile(fname);
    dict_prefix = 'train' if is_train else 'test'; 
    if (not file_exists):
        elbo_vals = np.nan;
        R2_vals = np.nan;
        KL_vals = np.nan;
        K = 1;
    else:
        X = np.load(fname);
        check_rate = X['check_rate'];
        elbos = X['%s_elbos' % dict_prefix];
        R2s = X['%s_R2s' % dict_prefix];
        max_checks = X['%s_R2s' % dict_prefix].shape[0];
        opt_ind = max_checks-1;
        K = X['%s_R2s' % dict_prefix].shape[1];
        elbo_vals = X['%s_elbos' % dict_prefix][opt_ind,:];
        R2_vals = X['%s_R2s' % dict_prefix][opt_ind,:];
        KL_vals = X['%s_KLs' % dict_prefix][opt_ind,:];
    elbo_nz_orig = np.sum(elbo_vals == 0);
    R2_nz_orig = np.sum(R2_vals == 0);
    KL_nz_orig = np.sum(KL_vals == 0);
    elbo_nans = np.sum(np.isnan(elbo_vals));
    R2_nans = np.sum(np.isnan(R2_vals));
    KL_nans = np.sum(np.isnan(KL_vals));
    if (elbo_nz_orig == K and R2_nz_orig == K):
        if (ind is None):
            ind = find_last_ind(elbos);
            ind_R2 = find_last_ind(R2s);
        elbo_vals = X['%s_elbos' % dict_prefix][ind-1,:];
        R2_vals = X['%s_R2s' % dict_prefix][ind-1,:];
        KL_vals = X['%s_KLs' % dict_prefix][ind-1,:];
        
    if (elbo_nz_orig > 0 or R2_nz_orig > 0 or elbo_nans > 0 or R2_nans > 0 or (not file_exists)):
        status = report_model_status(fname, elbo_nz_orig, R2_nz_orig, elbo_nans, R2_nans, file_exists);
    else:
        status = 0;
    return elbo_vals, R2_vals, KL_vals, status;

def log_fname(fname, status, status_lists):
    if status != 0:
        status_lists[status-1].append(fname);

def print_file_statuses(status_lists):
    not_started, in_progress, unstable = status_lists;
    text_start_ind = 43;
    print("Haven't started:");
    for fname in not_started:
        print(fname[text_start_ind:]);
    print('\n');
        
    print("Still running:");
    for fname in in_progress:
        print(fname[text_start_ind:]);  
    print('\n');

    print('Unstable:');
    for fname in unstable:
        print(fname[text_start_ind:]);
    print('\n');
    return None;


def load_dim_sweep(exp_fam, model, datadir, Ds, K, M, give_hint, max_iters, num_rs=10):
    num_Ds = len(Ds);
    if (model == 'EFN'):
        num_dists = K;
    else:
        num_dists = num_rs;
    elbos = np.zeros((num_Ds, num_dists));
    R2s = np.zeros((num_Ds, num_dists));
    KLs = np.zeros((num_Ds, num_dists));

    not_started = [];
    in_progress = [];
    unstable = [];
    status_lists = [not_started, in_progress, unstable];

    if (give_hint):
        give_inv_str = 'giveInv_';
    else:
        give_inv_str = '';

    for i in range(num_Ds):
        D = Ds[i];
        fam_class = family_from_str(exp_fam);
        family = fam_class(D);
        D_Z, ncons, num_param_net_inputs, num_Tx_inputs = family.get_efn_dims('eta', give_hint);
        #planar_flows = D;
        planar_flows = max(D, 20);
        flow_dict = get_flowdict(0, planar_flows, 0, 0);
        flowstring = get_flowstring(flow_dict);
        L = max(int(np.ceil(np.sqrt(D_Z))), 4);
        if (model == 'EFN'):
            fname = datadir + 'EFN_%s_stochasticEta_%sD=%d_K=%d_M=%d_flow=%s_L=%d_rs=%d/opt_info.npz' \
                                   % (exp_fam, give_inv_str, D, K, M, flowstring, L, 0);
            elbos[i, :], R2s[i,:], KLs[i,:], status = get_latest_diagnostics(fname, max_iters);
            log_fname(fname, status, status_lists);
        else:
            for rs in range(num_rs):
                if (model == 'NF1'):
                    fname = datadir + 'NF1/NF1_%s_D=%d_flow=%s_rs=%d/opt_info.npz' % (exp_fam, D, flowstring, rs+1);
                elif (model == 'EFN1'):
                    fname = datadir + 'EFN1/EFN_%s_fixedEta_%sD=%d_K=%d_M=%d_flow=%s_L=%d_rs=%d/opt_info.npz' \
                           % (exp_fam, give_inv_str, D, 1, M, flowstring, L, rs+1);
                elbos[i, rs], R2s[i,rs], KLs[i,rs], status = get_latest_diagnostics(fname, max_iters);
                log_fname(fname, status, status_lists);

    print_file_statuses(status_lists);

    return elbos, R2s, KLs;

def dim_sweep_df(Ds, model_strs, diagnostic_list):
    assert(len(model_strs) == len(diagnostic_list));
    num_models = len(model_strs);
    elbos_all = [];
    R2s_all = [];
    KLs_all = [];
    Ds_all = [];
    models = [];
    for i in range(num_models):
        elbos, R2s, KLs = diagnostic_list[i];
        elbos_vec = np.reshape(elbos.T, (np.prod(elbos.shape),));
        R2s_vec = np.reshape(R2s.T, (np.prod(R2s.shape),));
        KLs_vec = np.reshape(KLs.T, (np.prod(KLs.shape),));

        Ds_mat = np.tile(np.expand_dims(np.array(Ds), 1), (1, elbos.shape[1]));
        Ds_vec = np.reshape(Ds_mat.T, (np.prod(Ds_mat.shape),));

        elbo_isnan = np.isnan(elbos_vec)
        elbo_isnan = np.isnan(elbos_vec)
        R2_isnan = np.isnan(R2s_vec);
        R2_isnegative = R2s_vec < 0;
        remove_inds = np.logical_or(elbo_isnan, np.logical_or(R2_isnan, R2_isnegative));

        elbos_vec = elbos_vec[~remove_inds];
        R2s_vec = R2s_vec[~remove_inds];
        KLs_vec = KLs_vec[~remove_inds];
        Ds_vec = Ds_vec[~remove_inds];

        elbos_all.append(elbos_vec);
        R2s_all.append(R2s_vec);
        KLs_all.append(KLs_vec);
        Ds_all.append(Ds_vec);
        models.extend(elbos_vec.shape[0]*[model_strs[i]]);

    elbos = np.concatenate(elbos_all, 0);
    R2s = np.concatenate(R2s_all, 0);
    KLs = np.concatenate(KLs_all, 0);
    Ds = np.concatenate(Ds_all, 0);

    d = {"elbo":elbos, "R2":R2s, "KL":KLs, "D":Ds, "model":models};
    df = pd.DataFrame.from_dict(d);
    return df;

def EFN_model_df(id_name, id_labels, param_mat_lists, param_labels, diagnostic_list):
    assert(len(id_labels) == len(diagnostic_list));
    num_ids = len(id_labels);
    num_param_mats = len(param_labels);

    elbos_all = [];
    R2s_all = [];
    KLs_all = [];
    params_all = [];
    for i in range(num_param_mats):
        params_all.append([]);

    ids = [];
    for i in range(num_ids):
        elbos, R2s, KLs = diagnostic_list[i];
        K = elbos.shape[1];

        elbos_vec = np.reshape(elbos, (np.prod(elbos.shape),));
        R2s_vec = np.reshape(R2s, (np.prod(R2s.shape),));
        KLs_vec = np.reshape(KLs, (np.prod(KLs.shape),));

        param_vecs = [];
        for j in range(num_param_mats):
            param_mat_list = param_mat_lists[j];
            param_mat_ij = param_mat_list[i];
            param_vecs.append(np.reshape(param_mat_ij, (np.prod(param_mat_ij.shape),)));

        elbo_isnan = np.isnan(elbos_vec)
        elbo_isnan = np.isnan(elbos_vec)
        R2_isnan = np.isnan(R2s_vec);
        #R2_isnonpositive = R2s_vec <= 0;
        #remove_inds = np.logical_or(elbo_isnan, np.logical_or(R2_isnan, R2_isnonpositive));
        remove_inds = np.logical_or(elbo_isnan, R2_isnan);

        elbos_vec = elbos_vec[~remove_inds];
        R2s_vec = R2s_vec[~remove_inds];
        KLs_vec = KLs_vec[~remove_inds];
        for j in range(num_param_mats):
            param_vecs[j] = param_vecs[j][~remove_inds];

        elbos_all.append(elbos_vec);
        R2s_all.append(R2s_vec);
        KLs_all.append(KLs_vec);
        for j in range(num_param_mats):
            params_all[j].append(param_vecs[j]);

        ids.extend(elbos_vec.shape[0]*[id_labels[i]]);

    elbos = np.concatenate(elbos_all, 0);
    R2s = np.concatenate(R2s_all, 0);
    KLs = np.concatenate(KLs_all, 0);

    params = [];
    for i in range(num_param_mats):
        params.append(np.concatenate(params_all[i], 0));

    d = {"elbo":elbos, "R2":R2s, "KL":KLs, id_name:ids};
    for i in range(num_param_mats):
        d.update({param_labels[i]:params[i]});

    df = pd.DataFrame.from_dict(d);
    return df, d;


def load_V1_events(monkey, SNR_thresh=1.5, FR_thresh=1.0):
    spike_dir = '/Users/sbittner/Documents/efn/efn/data/pvc11/data_and_scripts/spikes_gratings/'
    fname = spike_dir + 'data_monkey%d_gratings.mat' % monkey;
    npzfile = sio.loadmat(fname);

    data = npzfile['data'];
    events = np.array(data['EVENTS'][0,0]);

    # cut by SNR thresh
    SNR = data['SNR'][0,0];
    keep_neurons = SNR > SNR_thresh;
    events = events[keep_neurons[:,0]];

    # cut by mean FR thresh
    num_neurons, num_oris, num_trials = events.shape;
    FRs = np.zeros((num_neurons, num_oris, num_trials))
    for i in range(num_neurons):
        for j in range(num_oris):
            for k in range(num_trials):
                events_ijk = events[i,j,k];
                inds = np.logical_and(0.28 <= events_ijk, events_ijk < 1.28);
                events[i,j,k] = events_ijk[inds] - 0.28;
                FRs[i,j,k] = np.count_nonzero(inds)/1.0;
    mean_FRs = np.mean(np.mean(FRs, 2), 1);
    keep_neurons = mean_FRs > FR_thresh;
    events = events[keep_neurons];

    return events;

def plotContourNormal(mu, Sigma, n_plot):
    sns.set_palette(sns.color_palette("Set1", n_colors=8, desat=.6))
    sns.set_style("whitegrid")
    sns.set_style("ticks")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
    sbpal = sns.color_palette()
    
    #width = 21;
    #height = 10;
    dist = multivariate_normal(mu, Sigma);
    spread1 = 3.5*Sigma[0,0];
    spread2 = 3.5*Sigma[1,1];
    x1 = np.linspace(mu[0]-spread1,mu[0]+spread1,n_plot)
    x2 = np.linspace(mu[1]-spread2,mu[1]+spread2,n_plot)
    Z = np.zeros([n_plot,n_plot])
    for i in range(n_plot):
        for j in range(n_plot):
            Z[i,j] = dist.pdf([x1[i],x2[j]])
    mycm = sns.light_palette("navy", as_cmap=True)
    mycm.set_under('w')
    plt.imshow(Z,extent=(mu[0]-spread1,mu[0]+spread1,mu[1]-spread2,mu[1]+spread2),cmap=mycm,origin='lower')
    plt.axis("off")
    norm_sample = dist.rvs(200)
    xns = norm_sample[:,0]
    yns = norm_sample[:,1]
    plt.scatter(xns,yns,c=sbpal[0])
    #plt.contourf(x,x,Z,cmap=mycm)
    plt.title("normal")
    plt.axis("equal")
    plt.tight_layout()
    plt.show();
    return fig;

corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

midpoints = [(corners[(i + 1) % 3] + corners[(i + 2) % 3]) / 2.0 \
             for i in range(3)]
def xy2bc(xy, tol=1.e-3):
    '''Converts 2D Cartesian coordinates to barycentric.'''
    s = [(corners[i] - midpoints[i]).dot(xy - midpoints[i]) / 0.75 \
         for i in range(3)]
    return np.clip(s, tol, 1.0 - tol)

def bc2xy(bc, tol=1.e-3):
    '''converts 3d barycentric coords to 2d cartesian.'''
    return corners.T.dot(bc)

class Dirichlet(object):
    def __init__(self, alpha):
        from math import gamma
        from operator import mul
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / \
                     reduce(mul, [gamma(a) for a in self._alpha])
    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        from operator import mul
        return self._coef * reduce(mul, [xx ** (aa - 1)
                                         for (xx, aa)in zip(x, self._alpha)])

def draw_pdf_contours(dist, subdiv=5, **kwargs):
    import math
    mycm = sns.light_palette("navy", as_cmap=True)
    mycm.set_under('w')
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]
    maxp = np.max(pvals);
    minp = np.min(pvals);
    step = ((maxp+1) - minp)/1000.0;
    contour_levels = np.arange(minp, maxp+1, step);
    plt.tricontourf(trimesh, pvals, contour_levels, cmap=mycm)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')


def plotContourTruncatedNormal(mu, Sigma, xlim, ylim, n_plot, scatter=True):
    sns.set_palette(sns.color_palette("Set1", n_colors=8, desat=.6))
    sns.set_style("whitegrid")
    sns.set_style("ticks")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
    sbpal = sns.color_palette()
    
    #width = 21;
    #height = 10;
    dist = multivariate_normal(mu, Sigma);
    x1 = np.linspace(0,xlim,n_plot)
    x2 = np.linspace(0,ylim,n_plot)
    Z = np.zeros([n_plot,n_plot])
    for i in range(n_plot):
        for j in range(n_plot):
            Z[i,j] = dist.pdf([x1[i],x2[j]]);
    mycm = sns.light_palette("navy", as_cmap=True)
    mycm.set_under('w')
    zmin = np.min(Z);
    zmax = np.max(Z);
    nc = 1000.0;
    z_contours = sattrend(zmin, zmax+(1.0/nc)*(zmax-zmin), nc, 2);
    plt.contourf(Z.T,levels=z_contours,cmap=mycm)
    #plt.axis("off")
    if (scatter):
        norm_sample = dist.rvs(100)
        xns = norm_sample[:,0]
        yns = norm_sample[:,1]
        plt.scatter(xns,yns,c=sbpal[0])


def exptrend(v1, v2, n, a):
    A = v2-v1;
    x = np.arange(0,1,1.0/n);
    y = np.exp(a*x) - 1;
    y = (A*y / np.max(y)) + v1;
    return y;

def sattrend(v1, v2, n, a):
    A = v2-v1;
    x = np.arange(0,1,1.0/n);
    y = 1 - np.exp(-a*x);
    y = (A*y / np.max(y)) + v1;
    return y;


def errorBars(x, y, err, legendstrs, color_palette=sns.xkcd_palette(colors)):
    fontsize = 16;
    num_trends = y.shape[0];
    xlen = x.shape[0];
    sizes = 40*np.ones((xlen,));
    for i in range(num_trends):
        color = np.tile(np.array([color_palette[i]]), [xlen, 1]);
        plt.scatter(x, y[i,:], sizes, c=color);
    plt.legend(legendstrs, fontsize=fontsize);

    for i in range(num_trends):
        for j in range(xlen):
            plt.plot([x[j], x[j]], [y[i,j]-err[i,j], y[i,j]+err[i,j]], '-', c=color_palette[i], lw=2);
    return None;
def plotMefnTraining(exp_fam, R2s, KLs, X, log_P, params, check_rate, iters, titlestr):
    fontsize = 14;
    num_checks = iters // check_rate;
    print('R2s shape', R2s.shape);
    K_eta = R2s.shape[1];
    its = check_rate*np.arange(1,num_checks+1, dtype=float);
    its = np.tile(np.expand_dims(its, 1), [1,K_eta]);
    its_vec = np.reshape(its, (num_checks*K_eta,));
    R2s_vec = np.reshape(R2s[:num_checks, :], (num_checks*K_eta,))
    KLs_vec = np.reshape(KLs[:num_checks, :], (num_checks*K_eta,))
    size = np.ones((num_checks*K_eta,));
    fig = plt.figure(figsize=(6,5));
        

    fig.add_subplot(2,2,1);
    plt.plot([np.min(its), np.max(its)], [1,1], 'tab:gray');
    plt.scatter(its_vec, R2s_vec, size,c='k');
    plt.legend(['goal', 'model']);
    plt.xlabel('iterations', fontsize=fontsize);
    plt.xlabel('iterations', fontsize=fontsize);
    plt.ylabel('R^2$', fontsize=fontsize)
    plt.ylim([-.2, 1.1]);

    fig.add_subplot(2,2,3);
    plt.plot([np.min(its), np.max(its)], [0,0], 'tab:gray');
    plt.scatter(its_vec, KLs_vec, size,c='k');
    plt.legend(['goal', 'model']);
    plt.xlabel('iterations', fontsize=fontsize);
    plt.ylabel('KL', fontsize=fontsize)

    # plot the distribution
    if (exp_fam == 'dirichlet'):
        fig.add_subplot(2,2,2);
        batch_size = X.shape[0];
        alpha = params['alpha'];
        D = alpha.shape[0];
        if (D == 3):
            X_true = np.random.dirichlet(alpha, (batch_size,));
            dist = dirichlet(np.float64(alpha));
            log_P_true = dist.logpdf(X_true.T);
            simplex.scatter(X, connect=False, c=log_P);
            plt.colorbar();
            plt.title('Q(X | theta, eta)', fontsize=fontsize)

            fig.add_subplot(2,2,4);
            simplex.scatter(X_true, connect=False, c=log_P_true);
            plt.colorbar();
            plt.title('true P(X | eta)');
            plt.suptitle(titlestr, fontsize=fontsize+2);
    return fig; 
def plotCategoricalPerformance(x, y, legendstrs=[], plottype='scatter', color_palette=sns.xkcd_palette(colors), dotsize = 5, shift=1):
    fontsize = 16;
    num_trends = len(y);
    xlen = x.shape[0];
    assert(xlen == y[0].shape[0]);
    Ns = [];
    for i in range(num_trends):
        Ns.append(y[i].shape[1]);
    maxN = max(Ns);
    
    sizes = dotsize*np.ones((1,));
    # set up legend
    if (len(legendstrs) > 0):
        for i in range(num_trends):
            color = np.tile(np.array([color_palette[i]]), [1, 1]);
            if (plottype == 'scatter'):
                plt.scatter(x[0], y[i][0,0], np.array([dotsize]), c=color);
            elif (plottype == 'errorBar'):
                plt.scatter(x[0], np.mean(y[i][0,:]), np.array([dotsize]), c=color);
        plt.legend(legendstrs, fontsize=fontsize);

    if (plottype == 'scatter'):
        xvals = np.zeros((num_trends*xlen*maxN,));
        yvals = np.zeros((num_trends*xlen*maxN,));
        colors = np.zeros((num_trends*xlen*maxN,3));
        sizes = dotsize*np.ones((num_trends*xlen*maxN,));
        ind = 0;
        sawzorn = False;
        for i in range(num_trends):
            if (plottype == 'scatter'):
                xshift_i = (i - (num_trends-1)/2)*shift;
            else:
                xshift_i = 0;
            N = Ns[i];
            for j in range(xlen):
                for n in range(N):
                    yval = y[i][j,n];
                    if (not sawzorn and (yval == 0 or np.isnan(yval))):
                        print('saw a zero or nan');
                        sawzorn = True;
                        continue;
                    yvals[ind] = yval
                    colors[ind,:] = np.array([color_palette[i]]);
                    xvals[ind] = x[j] + xshift_i
                    ind += 1;
        plt.scatter(xvals[:ind], yvals[:ind], sizes[:ind], c=colors[:ind]);

    elif (plottype == 'errorBar'):
        sizes = dotsize*np.ones((xlen,));
        means = np.zeros((num_trends, xlen));
        stds = np.zeros((num_trends, xlen));
        for i in range(num_trends):
            # make sure at the end there are no nans!
            means_i = np.nanmean(y[i], 1);
            means[i] = means_i;
            stds_i = np.nanstd(y[i], 1) / np.sqrt(Ns[i]);
            stds[i] = stds_i;
            plt.plot(x, means_i, '-', c=color_palette[i], lw=2);
        for i in range(num_trends):
            for j in range(xlen):
                plt.plot([x[j], x[j]], [means[i,j]-stds[i,j], means[i,j]+stds[i,j]], '-', c=color_palette[i], lw=2);
    
    return None;

def load_counts_spikes(monkey, neuron, ori):
    # get counts
    respdir = '/Users/sbittner/Documents/efn/efn/data/responses/';
    fname = respdir + 'spike_counts_monkey%d_neuron%d_ori%d.mat' % (monkey, neuron, ori);
    M = sio.loadmat(fname);
    counts = M['x'];
    
     # get spikes
    SNR_thresh = 1.5;
    FR_thresh = 1.0;
    events = load_V1_events(monkey, SNR_thresh, FR_thresh);
    spikes = events[neuron-1, ori-1,:];
    return counts, spikes;

def cut_trailing_spikes(events, D, T_s):
    ntrials = events.shape[0];
    spikes = [];
    t_end = D*T_s;
    for i in range(ntrials):
        _event_i = events[i];
        _event_i = [0.0] + _event_i[_event_i < t_end].tolist();
        spikes.append(_event_i);
    return spikes
    

def time_series_contour(Z, prctiles, T_s):
    num_prctiles = len(prctiles);
    is_odd = np.mod(num_prctiles,2)==1;
    if (not is_odd):
        print('Error: use odd number of symmetric percentiles');
        return None;
    mid_prctile_ind = num_prctiles // 2;
        
    
    T = Z.shape[1];
    zs = np.zeros((num_prctiles,T));
    for i in range(num_prctiles):
        for t in range(T):
            zs[i,t] = np.percentile(Z[:,t], prctiles[i]);
    

    t = np.linspace(0,(T-1)*T_s, T) + (T_s/2.0);
    for i in range(num_prctiles):
        if (i == mid_prctile_ind):
            plt.plot(t,zs[i], 'k');
        else:
            plt.plot(t,zs[i], 'k--');
    return None;
