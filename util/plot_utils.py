import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
#from dirichlet import simplex

def plotMefnTraining(exp_fam, R2s, KLs, X, log_P, params, check_rate, iters, titlestr):
    fontsize = 14;
    num_checks = iters // check_rate;
    print('R2s shape', R2s.shape);
    K_eta = R2s.shape[1];
    its = check_rate*np.arange(1,num_checks+1, dtype=float);
    print(its);
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