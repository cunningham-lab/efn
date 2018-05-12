import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.tri as tri
from scipy.stats import multivariate_normal
from functools import reduce

#from dirichlet import simplex
colors = ["pale red", "medium green", "windows blue", "amber", "dusty purple", "greyish", "faded green", "denim blue"];

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
    
def plotCategoricalPerformance(x, y, legendstrs=[], plottype='scatter', color_palette=sns.xkcd_palette(colors), dotsize = 5):
    fontsize = 16;
    num_trends = y.shape[0];
    xlen = x.shape[0];
    assert(xlen == y.shape[1]);
    N = y.shape[2];
    
    sizes = dotsize*np.ones((1,));
    # set up legend
    if (len(legendstrs) > 0):
        for i in range(num_trends):
            color = np.tile(np.array([color_palette[i]]), [1, 1]);
            if (plottype == 'scatter'):
                plt.scatter(x[0], y[i,0,0], np.array([dotsize]), c=color);
            elif (plottype == 'errorBar'):
                plt.scatter(x[0], np.mean(y[i,0,:]), np.array([dotsize]), c=color);
        plt.legend(legendstrs, fontsize=fontsize);

    if (plottype == 'scatter'):
        xvals = np.zeros((num_trends*xlen*N,));
        yvals = np.zeros((num_trends*xlen*N,));
        colors = np.zeros((num_trends*xlen*N,3));
        sizes = dotsize*np.ones((num_trends*xlen*N,));
        ind = 0;
        for n in range(N):
            for i in range(num_trends):
                for j in range(xlen):
                    yval = y[i,j,n];
                    if (yval == 0):
                        continue;
                    yvals[ind] = yval
                    colors[ind,:] = np.array([color_palette[i]]);
                    xvals[ind] = x[j] + (i-2)*.5;
                    ind += 1;
        plt.scatter(xvals[:ind], yvals[:ind], sizes[:ind], c=colors[:ind]);

    elif (plottype == 'errorBar'):
        means = np.mean(y, 2);
        stds = np.std(y, 2);
        sizes = dotsize*np.ones((xlen,));
        for i in range(num_trends):
            plt.plot(x, means[i], '-', c=color_palette[i], lw=2);
        for i in range(num_trends):
            for j in range(xlen):
                plt.plot([x[j], x[j]], [means[i,j]-stds[i,j], means[i,j]+stds[i,j]], '-', c=color_palette[i], lw=2);
    
    return None;

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
    plt.imshow(Z,extent=(mu[0]-spread1,mu[0]+spread1,mu[1]-spread2,mu[1]+spread2),cmap=mycm,origin='lower',vmin=0.000325)
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
    plt.tricontourf(trimesh, pvals, np.arange(0,15,0.1), cmap=mycm)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')
    plt.show();


def plotContourTruncatedNormal(mu, Sigma, xlim, ylim, n_plot, title='truncated normal', scatter=True):
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
            Z[i,j] = dist.pdf([x1[i],x2[j]]);
    mycm = sns.light_palette("navy", as_cmap=True)
    mycm.set_under('w')
    plt.imshow(Z.T,extent=(mu[0]-spread1,mu[0]+spread1,mu[1]-spread2,mu[1]+spread2),cmap=mycm,origin='lower',vmin=0.000325)
    #plt.axis("off")
    if (scatter):
        norm_sample = dist.rvs(100)
        xns = norm_sample[:,0]
        yns = norm_sample[:,1]
        plt.scatter(xns,yns,c=sbpal[0])
    #plt.contourf(x,x,Z,cmap=mycm)
    plt.title(title)
    #plt.tight_layout()
    plt.xlim([0, xlim]);
    plt.ylim([0, ylim]);



