import tensorflow as tf
import numpy as np
import scipy.stats
from scipy.special import gammaln, psi
import scipy.io as sio
from itertools import compress
from lib.tf_util.tf_util import count_layer_params
from lib.tf_util.stat_util import (
    truncated_multivariate_normal_rvs,
    get_GP_Sigma,
    drawPoissonCounts,
    get_sampler_func,
)
from lib.tf_util.flows import (
    SimplexBijectionLayer,
    CholProdLayer,
    SoftPlusLayer,
    ShiftLayer,
)


class Family:
    """Base class for exponential families.
	
	Exponential families differ in their sufficient statistics, base measures, supports,
	and therefore their natural parametrization.  Children of this class provide a set 
	of useful methods for learning particular exponential family models.

	Attributes:
		D (int): Dimensionality of the exponential family.
		D_Z (int): Dimensionality of the density network.
		T (int): Number of time points.
		num_suff_stats (int): Dimensionality of sufficient statistics vector.
		num_T_z_inputs (int): Number of param-dependent inputs to suff stat comp
		                      (only used in HierarchicalDirichlet).
		constant_base_measure (bool): True if base measure is sample independent.
		has_log_p (bool): True if a tractable form for sample log density is known.
		eta_dist (dict): Specifies the prior on the natural parameter, eta.
		eta_sampler (function): Returns a single sample from the eta prior.

	"""

    def __init__(self, D, T=1, eta_dist=None):
        """Family constructor.

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
			eta_dist (dict): Specifies the prior on the natural parameter, eta.

		"""

        self.D = D
        self.T = T
        self.num_T_z_inputs = 0
        self.constant_base_measure = True
        self.has_log_p = False
        if eta_dist is not None:
            self.eta_dist = eta_dist
        else:
            self.eta_dist = self.default_eta_dist()
        self.eta_sampler = get_sampler_func(self.eta_dist, self.D)

    def map_to_support(self, layers, num_theta_params):
        """Augment density network with bijective mapping to support."""
        return layers, num_theta_params

    def compute_suff_stats(self, Z, Z_by_layer, T_z_input):
        """Compute sufficient statistics of density network samples."""
        raise NotImplementedError()

    def compute_mu(self, params):
        """Compute the mean parameterization (mu) given the mean parameters."""
        raise NotImplementedError()

    def center_suff_stats_by_mu(self, T_z, mu):
        """Center sufficient statistics by the mean parameters mu."""
        return T_z - tf.expand_dims(tf.expand_dims(mu, 0), 1)

    def compute_log_base_measure(self, Z):
        """Compute log base measure of density network samples."""
        raise NotImplementedError()

    def default_eta_dist(self,):
        """Construct default eta prior."""
        raise NotImplementedError()

    def draw_etas(self, K, param_net_input_type="eta", give_hint=False):
        """Samples K times from eta prior and sets up parameter network input."""
        raise NotImplementedError()

    def mu_to_eta(self, params, param_net_input_type="eta", give_hint="False"):
        """Maps mean parameters (mu) of distribution to canonical parameters (eta)."""
        raise NotImplementedError()

    def mu_to_T_z_input(self, params):
        """Maps mean parameters (mu) of distribution to suff stat computation input.
		   (only necessary for HierarchicalDirichlet)

		Args:
			params (dict): Mean parameters.

		Returns:
			T_z_input (np.array): Param-dependent input.

		"""

        T_z_input = np.array([])
        return T_z_input

    def get_efn_dims(self, param_net_input_type="eta", give_hint=False):
        """Returns EFN component dimensionalities for the family."""
        raise NotImplementedError()

    def log_p(self, Z, params):
        """Computes log probability of Z given params."""
        raise NotImplementedError()

    def batch_diagnostics(
        self,
        K,
        sess,
        feed_dict,
        Z,
        log_p_z,
        elbos,
        R2s,
        eta_draw_params,
        checkEntropy=False,
    ):
        """Returns ELBOs, r^2s, and KL divergences of K distributions of family.

		Args:
			K (int): Number of distributions.
			sess (tf session): Running tf session.
			feed_dict (dict): Contains Z0, eta, param_net_input, and T_z_input.
			Z (log_h_z): Density network samples.
			log_p_z (tf.Tensor): Log probabilities of Z.
			elbos (tf.Tensor): ELBOs for each distribution.
			R2s (tf.Tensor): r^2s for each distribution
			eta_draw_params (list): Contains mean parameters of each distribution.
			check_entropy (bool): Print model entropy relative to true entropy.

		Returns:
			_elbos (np.array): Approximate ELBO for each distribution.
			_R2s (np.array): Approximate r^2s for each distribution.
			KLs (np.array): Approximate KL divergence for each distribution.
			_X (np.array): Density network samples.

		"""

        _Z, _log_p_z, _elbos, _R2s = sess.run([Z, log_p_z, elbos, R2s], feed_dict)
        KLs = []
        for k in range(K):
            if (self.has_log_p):
                log_p_z_k = _log_p_z[k, :]
                Z_k = _Z[k, :, :, 0]
                params_k = eta_draw_params[k]
                KL_k = self.approx_KL(log_p_z_k, Z_k, params_k)
                KLs.append(KL_k)
                if checkEntropy:
                    self.check_entropy(log_p_z_k, params_k)
            else:
                KLs.append(np.nan);
        return np.array(_elbos), np.array(_R2s), np.array(KLs), _Z

    def approx_KL(self, log_Q, Z, params):
        """Approximate KL(Q || P).

		Args:
			log_Q (np.array): log prob of density network samples.
			Z (np.array): Density network samples.
			params (dict): Mean parameters of target distribution.

		Returns:
			KL (np.float): KL(Q || P)
			
		"""

        log_P = self.log_p_np(Z, params)
        KL = np.mean(log_Q - log_P)
        return KL

    def approx_entropy(self, log_Q):
        """Approximates entropy of the sampled distribution.

		Args:
			log_Q (np.array): log probability of Q

		Returns:
			H (np.float): approximate entropy of Q
		"""

        return np.mean(-log_Q)

    def true_entropy(self, params):
        """Calculates true entropy of the distribution from mean parameters."""
        return np.nan

    def check_entropy(self, log_Q, params):
        """Prints entropy of approximate distribution relative to target distribution.

		Args:
			log_Q (np.array): log probability of Q.
			params (dict): Mean parameters of P.
		"""

        approxH = self.approx_entropy(log_Q)
        trueH = self.true_entropy(params)
        if not np.isnan(trueH):
            print("model entropy / true entropy")
            print("%.2E / %.2E" % (approxH, trueH))
        else:
            print("model entropy")
            print("%.2E" % approxH)
        return None


class PosteriorFamily(Family):
    """Base class for posterior-inference exponential families.
	
	When the likelihood of a bayesian model has exoponential family form and is closed 
	under sampling, we can learn the posterior-inference exponential family.  See section
	A.2 of the efn code docs.

	Attributes:
		D (int): Dimensionality of the exponential family.
		T (int): Number of time points.
		D_Z (int): Dimensionality of the density network.
		num_T_z_inputs (int): Number of param-dependent inputs to suff stat comp
		                      (only used in HierarchicalDirichlet).
		constant_base_measure (bool): True if base measure is sample independent.
		has_log_p (bool): True if a tractable form for sample log density is known.
		eta_dist (dict): Specifies the prior on the natural parameter, eta.
		eta_sampler (function): Returns a single sample from the eta prior.
		num_prior_suff_stats (int): Number of suff stats that come from prior.
		num_likelihood_suff_stats (int): " " from likelihood.
		num_suff_stats (int): Total number of suff stats.

	"""

    def __init__(self, D, T=1, eta_dist=None):
        """posterior family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
			eta_dist (dict): Specifies the prior on the natural parameter, eta.

		"""
        super().__init__(D, T, eta_dist)
        self.D_Z = None
        self.num_prior_suff_stats = None
        self.num_likelihood_suff_stats = None
        self.num_suff_stats = None

    def get_efn_dims(self, param_net_input_type="eta", give_hint=False):
        """Returns EFN component dimensionalities for the family.

		Args:
			param_net_input_type (str): Specifies input to param network.
				'eta':        Give full eta to parameter network.
				'prior':      Part of eta that is prior-dependent.
				'likelihood': Part of eta that is likelihood-dependent.
				'data':       The data itself.
			give_hint (bool): No hint implemented.

		Returns:
			D_Z (int): Dimensionality of density network.
			num_suff_stats: Dimensionality of eta.
			num_param_net_inputs: Dimensionality of parameter network input.
			num_T_z_inputs: Dimensionality of suff stat computation input.

		"""
        if give_hint:
            raise NotImplementedError()
        if param_net_input_type == "eta":
            num_param_net_inputs = self.num_suff_stats
        elif param_net_input_type == "prior":
            num_param_net_inputs = self.num_prior_suff_stats
        elif param_net_input_type == "likelihood":
            num_param_net_inputs = self.num_likelihood_suff_stats
        elif param_net_input_type == "data":
            num_param_net_inputs = self.D
        else:
            raise NotImplementedError()
        return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_z_inputs


class MultivariateNormal(Family):
    """Multivariate normal family."""

    def __init__(self, D, T=1, eta_dist=None):
        """Multivariate normal family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
			eta_dist (dict): Specifies the prior on the natural parameter, eta.

		"""
        super().__init__(D, T, eta_dist)
        self.name = "MultivariateNormal"
        self.D_Z = D
        self.num_suff_stats = int(D + D * (D + 1) / 2)
        self.has_log_p = True

    def compute_suff_stats(self, Z, Z_by_layer, T_z_input):
        """Compute sufficient statistics of density network samples.

		Args:
			Z (tf.Tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_z_input (tf.Tensor): Param-dependent input.

		Returns:
			T_z (tf.Tensor): Sufficient statistics of samples.

		"""
        Z_shape = tf.shape(Z)
        K = Z_shape[0]
        M = Z_shape[1]
        cov_con_mask = np.triu(np.ones((self.D, self.D), dtype=np.bool_), 0)
        T_z_mean = tf.reduce_mean(Z, 3)
        Z_KMTD = tf.transpose(Z, [0, 1, 3, 2])
        # samps x D
        ZZT_KMTDD = tf.matmul(tf.expand_dims(Z_KMTD, 4), tf.expand_dims(Z_KMTD, 3))
        T_z_cov_KMTDZ = tf.transpose(
            tf.boolean_mask(tf.transpose(ZZT_KMTDD, [3, 4, 0, 1, 2]), cov_con_mask),
            [1, 2, 3, 0],
        )
        T_z_cov = tf.reduce_mean(T_z_cov_KMTDZ, 2)
        T_z = tf.concat((T_z_mean, T_z_cov), axis=2)
        return T_z

    def compute_mu(self, params):
        """Compute the mean parameterization (mu) given the mean parameters.

        Args:
			params (dict): Mean parameters of distributions.

		Returns:
			mu (np.array): The mean parameterization vector of the exponential family.

		"""
        mu = params["mu"]
        Sigma = params["Sigma"]
        mu_mu = mu
        mu_Sigma = np.zeros((int(self.D * (self.D + 1) / 2)))
        ind = 0
        for i in range(self.D):
            for j in range(i, self.D):
                mu_Sigma[ind] = Sigma[i, j] + mu[i] * mu[j]
                ind += 1

        mu = np.concatenate((mu_mu, mu_Sigma), 0)
        return mu

    def compute_log_base_measure(self, Z):
        """Compute log base measure of density network samples.

		Args:
			X (tf.Tensor): Density network samples.

		Returns:
			log_h_z (tf.Tensor): Log base measure of samples.

		"""
        Z_shape = tf.shape(Z)
        K = Z_shape[0]
        M = Z_shape[1]
        log_h_z = -(self.D / 2) * np.log(2 * np.pi) * tf.ones((K, M), dtype=tf.float64)
        return log_h_z

    def default_eta_dist(self,):
        """Construct default eta prior.

		Returns:
			dist (dict): Default eta prior distribution.

		"""
        dist = {
            "family": "iso_mvn_and_iso_iw",
            "mu": np.zeros((self.D,)),
            "scale": 0.1,
            "df_fac": 5,
        }
        return dist

    def draw_etas(self, K, param_net_input_type="eta", give_hint=False):
        """Samples K times from eta prior and sets up parameter network input.

		Args:
			K (int): Number of distributions.
			param_net_input_type (str): Specifies input to param network.
				'eta':        Give full eta to parameter network.
			give_hint (bool): Feed in covariance cholesky if true.

		Returns:
			eta (np.array): K canonical parameters.
			param_net_inputs (np.array): K corresponding inputs to parameter network.
			T_z_input (np.array): K corresponding suff stat computation inputs.
			params (list): K corresponding mean parameterizations.
			
		"""
        _, _, num_param_net_inputs, _ = self.get_efn_dims(
            param_net_input_type, give_hint
        )
        eta = np.zeros((K, self.num_suff_stats))
        param_net_inputs = np.zeros((K, num_param_net_inputs))
        T_z_input = np.zeros((K, self.num_T_z_inputs))
        params = []
        for k in range(K):
            mu_k, Sigma_k = self.eta_sampler()
            params_k = {"mu": mu_k, "Sigma": Sigma_k}
            params.append(params_k)
            eta[k, :], param_net_inputs[k, :] = self.mu_to_eta(
                params_k, param_net_input_type, give_hint
            )
            T_z_input[k, :] = self.mu_to_T_z_input(params_k)
        return eta, param_net_inputs, T_z_input, params

    def mu_to_eta(self, params, param_net_input_type="eta", give_hint="False"):
        """Maps mean parameters (mu) of distribution to canonical parameters (eta).

		Args:
			params (dict): Mean parameters of distribution.
			param_net_input_type (str): Specifies input to param network.
				'eta':        Give full eta to parameter network.
			give_hint (bool): Feed in covariance cholesky if true.

		Returns:
			eta (np.array): Canonical parameters.
			param_net_input (np.array): Corresponding input to parameter network.

		"""

        if not param_net_input_type == "eta":
            raise NotImplementedError()
        mu = params["mu"]
        Sigma = params["Sigma"]
        cov_con_inds = np.triu_indices(self.D_Z, 0)
        upright_tri_inds = np.triu_indices(self.D_Z, 1)
        chol_inds = np.tril_indices(self.D_Z, 0)
        eta1 = np.float64(np.dot(np.linalg.inv(Sigma), np.expand_dims(mu, 1))).T
        eta2 = np.float64(-np.linalg.inv(Sigma) / 2)
        # by using the minimal representation, we need to multiply eta by two
        # for the off diagonal elements
        eta2[upright_tri_inds] = 2 * eta2[upright_tri_inds]
        eta2_minimal = eta2[cov_con_inds]
        eta = np.concatenate((eta1[0], eta2_minimal))

        if give_hint:
            L = np.linalg.cholesky(Sigma)
            chol_minimal = L[chol_inds]
            # param_net_input = np.concatenate((eta, chol_minimal));
            # param_net_input = np.concatenate((eta, mu, chol_minimal)); # add mu as well
            param_net_input = np.concatenate((mu, chol_minimal))
            # actually, no eta
        else:
            param_net_input = eta
        return eta, param_net_input

    def get_efn_dims(self, param_net_input_type="eta", give_hint=False):
        """Returns EFN component dimensionalities for the family.

		Args:
			param_net_input_type (str): Specifies input to param network.
				'eta':        Give full eta to parameter network.
			give_hint (bool): Feed in covariance cholesky and mean if true.

		Returns:
			D_Z (int): Dimensionality of density network.
			num_suff_stats: Dimensionality of eta.
			num_param_net_inputs: Dimensionality of parameter network input.
			num_T_z_inputs: Dimensionality of suff stat computation input.

		"""

        if not param_net_input_type == "eta":
            raise NotImplementedError()

        if give_hint:
            # num_param_net_inputs = int(2*self.D + self.D*(self.D+1));
            num_param_net_inputs = int(self.D + self.D * (self.D + 1) / 2)
        else:
            num_param_net_inputs = int(self.D + self.D * (self.D + 1) / 2)
        return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_z_inputs

    def log_p(self, Z, params):
        """Computes log probability of Z given params.

		Args:
			Z (tf.Tensor): Density network samples
			params (dict): Mean parameters of distribution.

		Returns:
			log_p (np.array): Ground truth probability of X given params.

		"""

        mu = params["mu"]
        Sigma = params["Sigma"]
        dist = tf.contrib.distributions.MultivariateNormalFullCovariance(
            loc=mu, covariance_matrix=Sigma
        )
        # dist = scipy.stats.multivariate_normal(mean=mu, cov=Sigma);
        assert self.T == 1
        log_p_z = dist.log_prob(Z[:, :, :, 0])
        return log_p_z

    def log_p_np(self, X, params):
        """Computes log probability of X given params.

		Args:
			X (np.array): Density network samples.
			params (dict): Mean parameters of distribution.

		Returns:
			log_p (np.array): Ground truth probability of X given params.

		"""

        mu = params["mu"]
        Sigma = params["Sigma"]
        dist = scipy.stats.multivariate_normal(mean=mu, cov=Sigma)
        assert self.T == 1
        log_p_x = dist.logpdf(X)
        return log_p_x

    def true_entropy(self, params):
        """Calculates true entropy of the distribution from mean parameters.

		Args:
			params (dict): Mean parameters of distribution.

		Returns:
			H_true (np.float): True distribution entropy.

		"""

        mu = params["mu"]
        Sigma = params["Sigma"]
        dist = scipy.stats.multivariate_normal(mean=mu, cov=Sigma)
        H_true = dist.entropy()
        return H_true


class Dirichlet(Family):
    """Dirichlet family.

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		D_Z (int): dimensionality of the density network
		num_suff_stats (int): total number of suff stats
		num_T_z_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""

    def __init__(self, D, T=1, eta_dist=None):
        """dirichlet family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.

		"""
        super().__init__(D, T, eta_dist)
        self.name = "Dirichlet"
        self.D_Z = D - 1
        self.num_suff_stats = D
        self.constant_base_measure = False
        self.has_log_p = True

    def map_to_support(self, layers, num_theta_params):
        """Augment density network with bijective mapping to support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.

		"""
        support_layer = SimplexBijectionLayer()
        num_theta_params += count_layer_params(support_layer)
        layers.append(support_layer)
        return layers, num_theta_params

    def compute_suff_stats(self, Z, Z_by_layer, T_z_input):
        """Compute sufficient statistics of density network samples.

		Args:
			Z (tf.Tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_z_input (tf.Tensor): Param-dependent input.

		Returns:
			T_z (tf.Tensor): Sufficient statistics of samples.

		"""

        Z_shape = tf.shape(Z)
        K = Z_shape[0]
        M = Z_shape[1]
        log_Z = tf.log(Z)
        T_z_log = tf.reduce_mean(log_Z, 3)
        T_z = T_z_log
        return T_z

    def compute_mu(self, params):
        """Compute the mean parameterization (mu) given the mean parameters.

        Args:
			params (dict): Mean parameters of distributions.

		Returns:
			mu (np.array): The mean parameterization vector of the exponential family.

		"""
        alpha = params["alpha"]
        alpha_0 = np.sum(alpha)
        phi_0 = psi(alpha_0)
        mu = psi(alpha) - phi_0
        return mu

    def compute_log_base_measure(self, Z):
        """Compute log base measure of density network samples.

		Args:
			Z (tf.Tensor): Density network samples.

		Returns:
			log_h_z (tf.Tensor): Log base measure of samples.
			
		"""
        assert self.T == 1
        log_h_z = -tf.reduce_sum(tf.log(Z), [2])
        return log_h_z[:, :, 0]

    def default_eta_dist(self,):
        """Construct default eta prior.

		Returns:
			dist (dict): Default eta prior distribution.

		"""
        dist = {"family": "uniform", "a": 0.5, "b": 5.0}
        return dist

    def draw_etas(self, K, param_net_input_type="eta", give_hint=False):
        """Samples K times from eta prior and sets up parameter network input.

		Args:
			K (int): Number of distributions.
			param_net_input_type (str): Specifies input to param network.
				'eta':        Give full eta to parameter network.
			give_hint (bool): No hint implemented.

		Returns:
			eta (np.array): K canonical parameters.
			param_net_inputs (np.array): K corresponding inputs to parameter network.
			T_z_input (np.array): K corresponding suff stat computation inputs.
			params (list): K corresponding mean parameterizations.
			
		"""
        _, _, num_param_net_inputs, _ = self.get_efn_dims(
            param_net_input_type, give_hint
        )
        eta = np.zeros((K, self.num_suff_stats))
        param_net_inputs = np.zeros((K, num_param_net_inputs))
        T_z_input = np.zeros((K, self.num_T_z_inputs))
        params = []
        for k in range(K):
            alpha_k = self.eta_sampler()
            params_k = {"alpha": alpha_k}
            params.append(params_k)
            eta[k, :], param_net_inputs[k, :] = self.mu_to_eta(
                params_k, param_net_input_type, give_hint
            )
            T_z_input[k, :] = self.mu_to_T_z_input(params_k)
        return eta, param_net_inputs, T_z_input, params

    def mu_to_eta(self, params, param_net_input_type="eta", give_hint=False):
        """Maps mean parameters (mu) of distribution to canonical parameters (eta).

		Args:
			params (dict): Mean parameters of distribution.
			param_net_input_type (str): Specifies input to param network.
				'eta':        Give full eta to parameter network.
			give_hint (bool): No hint implemented.

		Returns:
			eta (np.array): Canonical parameters.
			param_net_input (np.array): Corresponding input to parameter network.
			
		"""

        if give_hint or (not param_net_input_type == "eta"):
            raise NotImplementedError()
        alpha = params["alpha"]
        eta = alpha
        param_net_input = alpha
        return eta, param_net_input

    def get_efn_dims(self, param_net_input_type="eta", give_hint=False):
        """Returns EFN component dimensionalities for the family.

		Args:
			param_net_input_type (str): Specifies input to param network.
				'eta':        Give full eta to parameter network.
			give_hint (bool): No hint implemented.

		Returns:
			D_Z (int): Dimensionality of density network.
			num_suff_stats: Dimensionality of eta.
			num_param_net_inputs: Dimensionality of parameter network input.
			num_T_z_inputs: Dimensionality of suff stat computation input.
			
		"""

        if give_hint or (not param_net_input_type == "eta"):
            raise NotImplementedError()
        num_param_net_inputs = self.D
        return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_z_inputs

    def log_p(self, Z, params):
        """Computes log probability of Z given params.

		Args:
			Z (tf.Tensor): Density network samples
			params (dict): Mean parameters of distribution.

		Returns:
			log_p (np.array): Ground truth probability of Z given params.
			
		"""

        alpha = params["alpha"]
        dist = tf.contrib.distributions.Dirichlet(alpha)
        assert self.T == 1
        log_p_z = dist.log_prob(Z[:, :, :, 0])
        return log_p_z

    def log_p_np(self, Z, params):
        """Computes log probability of Z given params.

		Args:
			Z (np.array): Density network samples.
			params (dict): Mean parameters of distribution.

		Returns:
			log_p (np.array): Ground truth probability of Z given params.

		"""
        nonzero_simplex_eps = 1e-32
        alpha = params["alpha"]
        dist = scipy.stats.dirichlet(np.float64(alpha))
        Z = np.float64(Z) + nonzero_simplex_eps
        Z = Z / np.expand_dims(np.sum(Z, 1), 1)
        log_p_z = dist.logpdf(Z.T)
        return log_p_z

    def true_entropy(self, params):
        """Calculates true entropy of the distribution from mean parameters.

		Args:
			params (dict): Mean parameters of distribution.

		Returns:
			H_true (np.float): True distribution entropy.

		"""

        alpha = params["alpha"]
        dist = scipy.stats.dirichlet(np.float64(alpha))
        H_true = dist.entropy()
        return H_true


class InvWishart(Family):
    """Inverse-Wishart family.

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		D_Z (int): dimensionality of the density network
		num_suff_stats (int): total number of suff stats
		num_T_z_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""

    def __init__(self, D, T=1, eta_dist=None, diag_eps=1e-4):
        """inv_wishart family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""

        self.sqrtD = int(np.sqrt(D))
        super().__init__(D, T, eta_dist)
        self.name = "InvWishart"
        self.D_Z = int(self.sqrtD * (self.sqrtD + 1) / 2)
        self.num_suff_stats = self.D_Z + 1
        self.has_log_p = True
        self.diag_eps = 1e-10

    def map_to_support(self, layers, num_theta_params):
        """Augment density network with bijective mapping to support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.

		"""
        support_layer = CholProdLayer()
        num_theta_params += count_layer_params(support_layer)
        layers.append(support_layer)
        return layers, num_theta_params

    def compute_suff_stats(self, Z, Z_by_layer, T_z_input):
        """Compute sufficient statistics of density network samples.

		Args:
			Z (tf.Tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_z_input (tf.Tensor): Param-dependent input.

		Returns:
			T_z (tf.Tensor): Sufficient statistics of samples.

		"""

        assert self.T == 1
        Z_shape = tf.shape(Z)
        K = Z_shape[0]
        M = Z_shape[1]
        cov_con_mask = np.triu(np.ones((self.sqrtD, self.sqrtD), dtype=np.bool_), 0)
        chol_mask = np.tril(np.ones((self.sqrtD, self.sqrtD), dtype=np.bool_), 0)

        Z_KMDsqrtDsqrtD = tf.reshape(Z, (K, M, self.sqrtD, self.sqrtD))
        Z_inv = tf.matrix_inverse(Z_KMDsqrtDsqrtD)
        T_z_inv = tf.transpose(
            tf.boolean_mask(tf.transpose(Z_inv, [2, 3, 0, 1]), cov_con_mask), [1, 2, 0]
        )
        # We already have the Chol factor from earlier in the graph
        # zchol = Z_by_layer[-2];
        # zchol_KMD_Z = zchol[:,:,:,0]; # generalize this for more time points
        Z_eigs = tf.self_adjoint_eigvals(Z_KMDsqrtDsqrtD)
        # zchol_KMD = tf.transpose(tf.boolean_mask(tf.transpose(zchol_KMsqrtDsqrtD, [2,3,0,1]), chol_mask), [1, 2, 0]);

        T_z_log_det = 2 * tf.reduce_sum(tf.log(Z_eigs), 2)
        T_z_log_det = tf.expand_dims(T_z_log_det, 2)
        T_z = tf.concat((T_z_inv, T_z_log_det), axis=2)
        return T_z

    def compute_log_base_measure(self, Z):
        """Compute log base measure of density network samples.

		Args:
			Z (tf.Tensor): Density network samples.

		Returns:
			log_h_z (tf.Tensor): Log base measure of samples.
			
		"""
        assert self.T == 1
        Z_shape = tf.shape(Z)
        K = Z_shape[0]
        M = Z_shape[1]
        log_h_z = tf.zeros((K, M), dtype=tf.float64)
        return log_h_z

    def default_eta_dist(self,):
        """Construct default eta prior.

		Returns:
			dist (dict): Default eta prior distribution.

		"""
        dist = {
            "family": "ui_and_iso_iw",
            "ui_dim": 1,
            "iw_dim": self.sqrtD,
            "a": 2 * self.sqrtD,
            "b": 3 * self.sqrtD,
            "df_fac": 100,
        }
        return dist

    def draw_etas(self, K, param_net_input_type="eta", give_hint=False):
        """Samples K times from eta prior and sets up parameter network input.

		Args:
			K (int): Number of distributions.
			param_net_input_type (str): Specifies input to param network.
				'eta':        Give full eta to parameter network.
			give_hint (bool): Feed in inverse Psi cholesky if True.

		Returns:
			eta (np.array): K canonical parameters.
			param_net_inputs (np.array): K corresponding inputs to parameter network.
			T_z_input (np.array): K corresponding suff stat computation inputs.
			params (list): K corresponding mean parameterizations.
			
		"""
        _, _, num_param_net_inputs, _ = self.get_efn_dims(
            param_net_input_type, give_hint
        )
        eta = np.zeros((K, self.num_suff_stats))
        param_net_inputs = np.zeros((K, num_param_net_inputs))
        T_z_input = np.zeros((K, self.num_T_z_inputs))

        params = []
        for k in range(K):
            m_k, Psi_k = self.eta_sampler()
            params_k = {"Psi": Psi_k, "m": m_k}
            params.append(params_k)
            eta[k, :], param_net_inputs[k, :] = self.mu_to_eta(
                params_k, param_net_input_type, give_hint
            )
            T_z_input[k, :] = self.mu_to_T_z_input(params_k)
        return eta, param_net_inputs, T_z_input, params

    def mu_to_eta(self, params, param_net_input_type="eta", give_hint=False):
        """Maps mean parameters (mu) of distribution to canonical parameters (eta).

		Args:
			params (dict): Mean parameters of distribution.
			param_net_input_type (str): Specifies input to param network.
				'eta':        Give full eta to parameter network.
			give_hint (bool): Feed in inverse Psi cholesky if True.

		Returns:
			eta (np.array): Canonical parameters.
			param_net_input (np.array): Corresponding input to parameter network.
			
		"""

        if not param_net_input_type == "eta":
            raise NotImplementedError()
        Psi = params["Psi"]
        m = params["m"]
        cov_con_inds = np.triu_indices(self.sqrtD, 0)
        upright_tri_inds = np.triu_indices(self.sqrtD, 1)
        eta1 = -Psi / 2.0
        eta1[upright_tri_inds] = 2 * eta1[upright_tri_inds]
        eta1_minimal = eta1[cov_con_inds]
        eta2 = np.array([-(m + self.sqrtD + 1) / 2.0])
        eta = np.concatenate((eta1_minimal, eta2))

        if give_hint:
            Psi_inv = np.linalg.inv(Psi)
            Psi_inv_minimal = Psi_inv[cov_con_inds]
            param_net_input = np.concatenate((eta, Psi_inv_minimal))
        else:
            param_net_input = eta
        return eta, param_net_input

    def get_efn_dims(self, param_net_input_type="eta", give_hint=False):
        """Returns EFN component dimensionalities for the family.

		Args:
			param_net_input_type (str): Specifies input to param network.
				'eta':        Give full eta to parameter network.
			give_hint (bool): Feed in inverse Psi cholesky if True.

		Returns:
			D_Z (int): Dimensionality of density network.
			num_suff_stats: Dimensionality of eta.
			num_param_net_inputs: Dimensionality of parameter network input.
			num_T_z_inputs: Dimensionality of suff stat computation input.
			
		"""

        if not param_net_input_type == "eta":
            raise NotImplementedError()

        if give_hint:
            num_param_net_inputs = 2 * self.D_Z + 1
        else:
            num_param_net_inputs = self.num_suff_stats

        return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_z_inputs

    def log_p_np(self, Z, params):
        """Computes log probability of Z given params.

		Args:
			Z (np.array): Density network samples.
			params (dict): Mean parameters of distribution.

		Returns:
			log_p (np.array): Ground truth probability of Z given params.

		"""

        batch_size = Z.shape[0]
        Psi = params["Psi"]
        m = params["m"]
        Z = np.reshape(Z, [batch_size, self.sqrtD, self.sqrtD])
        log_p_z = scipy.stats.invwishart.logpdf(
            np.transpose(Z, [1, 2, 0]), float(m), Psi
        )
        return log_p_z


class HierarchicalDirichlet(PosteriorFamily):
    """Hierarchical Dirichlet family.

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		D_Z (int): dimensionality of the density network
		num_prior_suff_stats (int): number of suff stats that come from prior
		num_likelihood_suff_stats (int): " " from likelihood
		num_suff_stats (int): total number of suff stats
		num_T_z_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""

    def __init__(self, D, T=1, eta_dist=None):
        """hierarchical_dirichlet family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""

        super().__init__(D, T, eta_dist)
        self.name = "HierarchicalDirichlet"
        self.D_Z = D - 1
        self.num_prior_suff_stats = D + 1
        self.num_likelihood_suff_stats = D + 1
        self.num_suff_stats = self.num_prior_suff_stats + self.num_likelihood_suff_stats
        self.num_T_z_inputs = 1

    def map_to_support(self, layers, num_theta_params):
        """Augment density network with bijective mapping to support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.

		"""
        support_layer = SimplexBijectionLayer()
        num_theta_params += count_layer_params(support_layer)
        layers.append(support_layer)
        return layers, num_theta_params

    def compute_suff_stats(self, Z, Z_by_layer, T_z_input):
        """Compute sufficient statistics of density network samples.

		Args:
			Z (tf.Tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_z_input (tf.Tensor): Param-dependent input.

		Returns:
			T_z (tf.Tensor): Sufficient statistics of samples.

		"""

        assert self.T == 1
        Z_shape = tf.shape(Z)
        K = Z_shape[0]
        M = Z_shape[1]
        logz = tf.log(Z[:, :, :, 0])
        const = -tf.ones((K, M, 1), tf.float64)
        beta = tf.expand_dims(T_z_input, 1)
        betaz = tf.multiply(beta, Z[:, :, :, 0])
        log_gamma_beta_z = tf.lgamma(betaz)
        log_gamma_beta = tf.lgamma(beta)
        log_Beta_beta_z = (
            tf.expand_dims(tf.reduce_sum(log_gamma_beta_z, 2), 2) - log_gamma_beta
        )
        T_z = tf.concat((logz, const, betaz, log_Beta_beta_z), 2)
        return T_z

    def compute_log_base_measure(self, Z):
        """Compute log base measure of density network samples.

		Args:
			Z (tf.Tensor): Density network samples.

		Returns:
			log_h_z (tf.Tensor): Log base measure of samples.
			
		"""
        assert self.T == 1
        Z_shape = tf.shape(Z)
        K = Z_shape[0]
        M = Z_shape[1]
        log_h_z = tf.zeros((K, M), dtype=tf.float64)
        return log_h_z

    def default_eta_dist(self,):
        """Construct default eta prior.

		Returns:
			dist (dict): Default eta prior distribution.

		"""
        dist = {
            "family": "dir_dir",
            "a_z": 0.5,
            "b_z": 5.0,
            "a_x": self.D,
            "b_x": 2 * self.D,
        }
        return dist

    def draw_etas(self, K, param_net_input_type="eta", give_hint=False):
        """Samples K times from eta prior and sets up parameter network input.

		Args:
			K (int): Number of distributions.
			param_net_input_type (str): Specifies input to param network.
				'eta':        Give full eta to parameter network.
				'prior':      Part of eta that is prior-dependent.
				'likelihood': Part of eta that is likelihood-dependent.
				'data':       The data itself.
			give_hint (bool): No hint implemented.

		Returns:
			eta (np.array): K canonical parameters.
			param_net_inputs (np.array): K corresponding inputs to parameter network.
			T_z_input (np.array): K corresponding suff stat computation inputs.
			params (list): K corresponding mean parameterizations.
			
		"""
        _, _, num_param_net_inputs, _ = self.get_efn_dims(
            param_net_input_type, give_hint
        )
        eta = np.zeros((K, self.num_suff_stats))
        param_net_inputs = np.zeros((K, num_param_net_inputs))
        T_z_input = np.zeros((K, self.num_T_z_inputs))
        Nmean = 10
        x_eps = 1e-16
        params = []
        for k in range(K):
            alpha_0_k = np.random.uniform(0.5, 5.0, (self.D,))
            beta_k = np.random.uniform(self.D, 2 * self.D)
            N = np.random.poisson(Nmean)
            # N = np.random.poisson(Nmean);
            dist1 = scipy.stats.dirichlet(alpha_0_k)
            z = dist1.rvs(1)
            dist2 = scipy.stats.dirichlet(beta_k * z[0])
            x = dist2.rvs(N).T
            x = x + x_eps
            x = x / np.expand_dims(np.sum(x, 0), 0)
            params_k = {"alpha_0": alpha_0_k, "beta": beta_k, "x": x, "z": z, "N": N}
            params.append(params_k)
            eta[k, :], param_net_inputs[k, :] = self.mu_to_eta(
                params_k, param_net_input_type, False
            )
            T_z_input[k, :] = self.mu_to_T_z_input(params_k)
        return eta, param_net_inputs, T_z_input, params

    def mu_to_eta(self, params, param_net_input_type="eta", give_hint=False):
        """Maps mean parameters (mu) of distribution to canonical parameters (eta).

		Args:
			params (dict): Mean parameters of distribution.
			param_net_input_type (str): Specifies input to param network.
				'eta':        Give full eta to parameter network.
				'prior':      Part of eta that is prior-dependent.
				'likelihood': Part of eta that is likelihood-dependent.
				'data':       The data itself.
			give_hint (bool): No hint implemented.

		Returns:
			eta (np.array): Canonical parameters.
			param_net_input (np.array): Corresponding input to parameter network.
			
		"""

        if give_hint:
            raise NotImplementedError()
        alpha_0 = params["alpha_0"]
        x = params["x"]
        N = params["N"]
        assert N == x.shape[1]

        log_Beta_alpha_0 = np.array(
            [np.sum(gammaln(alpha_0)) - gammaln(np.sum(alpha_0))]
        )
        sumlogx = np.sum(np.log(x), 1)

        eta_from_prior = np.concatenate((alpha_0 - 1.0, log_Beta_alpha_0), 0)
        eta_from_likelihood = np.concatenate((sumlogx, -np.array([N])), 0)
        eta = np.concatenate((eta_from_prior, eta_from_likelihood), 0)

        if param_net_input_type == "eta":
            param_net_input = eta
        elif param_net_input_type == "prior":
            param_net_input = eta_from_prior
        elif param_net_input_type == "likelihood":
            param_net_input = eta_from_likelihood
        elif param_net_input_type == "data":
            assert x.shape[1] == 1 and N == 1
            param_net_input = x.T
        return eta, param_net_input

    def mu_to_T_z_input(self, params):
        """Maps mean parameters (mu) of distribution to suff stat comp input.

		Args:
			params (dict): Mean parameters of distributions.

		Returns:
			T_z_input (np.array): Param-dependent input.
		"""

        beta = params["beta"]
        T_z_input = np.array([beta])
        return T_z_input


class DirichletMultinomial(PosteriorFamily):
    """Dirichlet-multinomial family.

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		D_Z (int): dimensionality of the density network
		num_prior_suff_stats (int): number of suff stats that come from prior
		num_likelihood_suff_stats (int): " " from likelihood
		num_suff_stats (int): total number of suff stats
		num_T_z_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""

    def __init__(self, D, T=1, eta_dist=None):
        """dirichlet_multinomial family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""

        super().__init__(D, T, eta_dist)
        self.name = "DirichletMultinomial"
        self.D_Z = D - 1
        self.num_prior_suff_stats = D + 1
        self.num_likelihood_suff_stats = D + 1
        self.num_suff_stats = self.num_prior_suff_stats + self.num_likelihood_suff_stats
        self.num_T_z_inputs = 0

    def map_to_support(self, layers, num_theta_params):
        """Augment density network with bijective mapping to support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.

		"""
        support_layer = SimplexBijectionLayer()
        num_theta_params += count_layer_params(support_layer)
        layers.append(support_layer)
        return layers, num_theta_params

    def compute_suff_stats(self, Z, Z_by_layer, T_z_input):
        """Compute sufficient statistics of density network samples.

		Args:
			Z (tf.Tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_z_input (tf.Tensor): Param-dependent input.

		Returns:
			T_z (tf.Tensor): Sufficient statistics of samples.

		"""

        assert self.T == 1
        Z_shape = tf.shape(Z)
        K = Z_shape[0]
        M = Z_shape[1]
        logz = tf.log(Z[:, :, :, 0])
        const = -tf.ones((K, M, 1), tf.float64)
        zeros = -tf.zeros((K, M, 1), tf.float64)
        T_z = tf.concat((logz, const, logz, zeros), 2)
        return T_z

    def compute_log_base_measure(self, Z):
        """Compute log base measure of density network samples.

		Args:
			Z (tf.Tensor): Density network samples.

		Returns:
			log_h_z (tf.Tensor): Log base measure of samples.
			
		"""
        assert self.T == 1
        Z_shape = tf.shape(Z)
        K = Z_shape[0]
        M = Z_shape[1]
        log_h_z = tf.zeros((K, M), dtype=tf.float64)
        return log_h_z

    def default_eta_dist(self,):
        """Construct default eta prior.

		Returns:
			dist (dict): Default eta prior distribution.

		"""
        dist = {"family": "dir_mult", "a_uniform": 0.5, "b": 5.0, "a_delta": 1}
        return dist

    def draw_etas(self, K, param_net_input_type="eta", give_hint=False):
        """Samples K times from eta prior and sets up parameter network input.

		Args:
			K (int): Number of distributions.
			param_net_input_type (str): Specifies input to param network.
				'eta':        Give full eta to parameter network.
				'prior':      Part of eta that is prior-dependent.
				'likelihood': Part of eta that is likelihood-dependent.
				'data':       The data itself.
			give_hint (bool): No hint implemented.

		Returns:
			eta (np.array): K canonical parameters.
			param_net_inputs (np.array): K corresponding inputs to parameter network.
			T_z_input (np.array): K corresponding suff stat computation inputs.
			params (list): K corresponding mean parameterizations.
			
		"""
        _, _, num_param_net_inputs, _ = self.get_efn_dims(
            param_net_input_type, give_hint
        )
        eta = np.zeros((K, self.num_suff_stats))
        param_net_inputs = np.zeros((K, num_param_net_inputs))
        T_z_input = np.zeros((K, self.num_T_z_inputs))
        N = 1
        x_eps = 1e-16
        params = []
        for k in range(K):
            alpha_0_k, x_k = self.eta_sampler()
            params_k = {"alpha_0": alpha_0_k, "x": x_k}
            params.append(params_k)
            eta[k, :], param_net_inputs[k, :] = self.mu_to_eta(
                params_k, param_net_input_type, False
            )
            T_z_input[k, :] = self.mu_to_T_z_input(params_k)
        return eta, param_net_inputs, T_z_input, params

    def mu_to_eta(self, params, param_net_input_type="eta", give_hint=False):
        """Maps mean parameters (mu) of distribution to canonical parameters (eta).

		Args:
			params (dict): Mean parameters of distribution.
			param_net_input_type (str): Specifies input to param network.
				'eta':        Give full eta to parameter network.
				'prior':      Part of eta that is prior-dependent.
				'likelihood': Part of eta that is likelihood-dependent.
				'data':       The data itself.
			give_hint (bool): No hint implemented.

		Returns:
			eta (np.array): Canonical parameters.
			param_net_input (np.array): Corresponding input to parameter network.
			
		"""

        if give_hint:
            raise NotImplementedError()
        alpha_0 = params["alpha_0"]
        x = params["x"]
        N = np.sum(x)

        log_Beta_alpha_0 = np.array(
            [np.sum(gammaln(alpha_0)) - gammaln(np.sum(alpha_0))]
        )

        eta_from_prior = np.concatenate((alpha_0 - 1.0, log_Beta_alpha_0), 0)
        eta_from_likelihood = np.concatenate((x[0, :], -np.array([N])), 0)
        eta = np.concatenate((eta_from_prior, eta_from_likelihood), 0)

        if param_net_input_type == "eta":
            param_net_input = eta
        elif param_net_input_type == "prior":
            param_net_input = eta_from_prior
        elif param_net_input_type == "likelihood":
            param_net_input = eta_from_likelihood
        elif param_net_input_type == "data":
            assert x.shape[1] == 1 and N == 1
            param_net_input = x.T
        return eta, param_net_input


class TruncatedNormalPoisson(PosteriorFamily):
    """Truncated normal Poisson family.

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		D_Z (int): dimensionality of the density network
		num_prior_suff_stats (int): number of suff stats that come from prior
		num_likelihood_suff_stats (int): " " from likelihood
		num_suff_stats (int): total number of suff stats
		num_T_z_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""

    def __init__(self, D, T=1, eta_dist=None):
        """truncated_normal_poisson family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""

        super().__init__(D, T, eta_dist)
        self.name = "TruncatedNormalPoisson"
        self.D_Z = D
        self.num_prior_suff_stats = int(D + D * (D + 1) / 2) + 1
        self.num_likelihood_suff_stats = D + 1
        self.num_suff_stats = self.num_prior_suff_stats + self.num_likelihood_suff_stats
        self.num_T_z_inputs = 0
        self.prior_family = MultivariateNormal(D, T)

    def map_to_support(self, layers, num_theta_params):
        """Augment density network with bijective mapping to support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.

		"""
        support_layer = SoftPlusLayer()
        num_theta_params += count_layer_params(support_layer)
        layers.append(support_layer)
        return layers, num_theta_params

    def compute_suff_stats(self, Z, Z_by_layer, T_z_input):
        """Compute sufficient statistics of density network samples.

		Args:
			Z (tf.Tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_z_input (tf.Tensor): Param-dependent input.

		Returns:
			T_z (tf.Tensor): Sufficient statistics of samples.

		"""

        assert self.T == 1
        Z_shape = tf.shape(Z)
        K = Z_shape[0]
        M = Z_shape[1]
        T_z_prior = self.prior_family.compute_suff_stats(Z, Z_by_layer, T_z_input)
        const = -tf.ones((K, M, 1), tf.float64)
        logz = tf.log(Z[:, :, :, 0])
        sumz = tf.expand_dims(tf.reduce_sum(Z[:, :, :, 0], 2), 2)
        T_z = tf.concat((T_z_prior, const, logz, sumz), 2)
        return T_z

    def compute_log_base_measure(self, Z):
        """Compute log base measure of density network samples.

		Args:
			Z (tf.Tensor): Density network samples.

		Returns:
			log_h_z (tf.Tensor): Log base measure of samples.
			
		"""
        assert self.T == 1
        Z_shape = tf.shape(Z)
        K = Z_shape[0]
        M = Z_shape[1]
        log_h_z = tf.zeros((K, M), dtype=tf.float64)
        return log_h_z

    def draw_etas(self, K, param_net_input_type="eta", give_hint=False):
        """Samples K times from eta prior and sets up parameter network input.

		Args:
			K (int): Number of distributions.
			param_net_input_type (str): specifies input to param network
				'eta':        give full eta to parameter network
				'prior':      part of eta that is prior-dependent
				'likelihood': part of eta that is likelihood-dependent
				'data':       the data itself
			give_hint (bool): Feed in prior covariance cholesky if true.

		Returns:
			eta (np.array): K canonical parameters.
			param_net_inputs (np.array): K corresponding inputs to parameter network.
			T_z_input (np.array): K corresponding suff stat computation inputs.
			params (list): K corresponding mean parameterizations.
			
		"""
        _, _, num_param_net_inputs, _ = self.get_efn_dims(
            param_net_input_type, give_hint
        )
        eta = np.zeros((K, self.num_suff_stats))
        param_net_inputs = np.zeros((K, num_param_net_inputs))
        T_z_input = np.zeros((K, self.num_T_z_inputs))
        nneurons = 83
        noris = 12
        Ts = 0.02
        mean_FR = 0.1169
        var_FR = 0.0079
        mu = mean_FR * np.ones((self.D_Z,))
        tau = 0.025
        # Sigma = var_FR*np.eye(self.D_Z);
        Sigma = var_FR * get_GP_Sigma(tau, self.D_Z, Ts)
        params = []
        data_sets = np.random.choice(nneurons * noris, K, False)
        for k in range(K):
            neuron = (data_sets[k] // noris) + 1
            ori = np.mod(data_sets[k], noris) + 1
            M = sio.loadmat(datadir + "spike_counts_neuron%d_ori%d.mat" % (neuron, ori))
            x = M["x"][:, : self.D_Z].T
            N = x.shape[1]
            params_k = {
                "mu": mu,
                "Sigma": Sigma,
                "x": x,
                "N": N,
                "monkey": 1,
                "neuron": neuron,
                "ori": ori,
            }
            params.append(params_k)
            eta[k, :], param_net_inputs[k, :] = self.mu_to_eta(
                params_k, param_net_input_type, give_hint
            )
            T_z_input[k, :] = self.mu_to_T_z_input(params_k)
        return eta, param_net_inputs, T_z_input, params

    def mu_to_eta(self, params, param_net_input_type="eta", give_hint=False):
        """Maps mean parameters (mu) of distribution to canonical parameters (eta).

		Args:
			params (dict): Mean parameters of distribution.
			param_net_input_type (str): specifies input to param network
				'eta':        give full eta to parameter network
				'prior':      part of eta that is prior-dependent
				'likelihood': part of eta that is likelihood-dependent
				'data':       the data itself
			give_hint (bool): Feed in prior covariance cholesky if true.

		Returns:
			eta (np.array): Canonical parameters.
			param_net_input (np.array): Corresponding input to parameter network.
			
		"""

        mu = params["mu"]
        Sigma = params["Sigma"]
        x = params["x"]
        N = params["N"]
        assert N == x.shape[1]

        alpha, alpha_param_net_input = self.prior_family.mu_to_eta(
            params, param_net_input_type, give_hint
        )
        mu = np.expand_dims(mu, 1)
        log_A_0 = 0.5 * (
            np.dot(mu.T, np.dot(np.linalg.inv(Sigma), mu))
            + np.log(np.linalg.det(Sigma))
        )
        sumx = np.sum(x, 1)

        eta_from_prior = np.concatenate((alpha, log_A_0[0]), 0)
        eta_from_likelihood = np.concatenate((sumx, -np.array([N])), 0)
        eta = np.concatenate((eta_from_prior, eta_from_likelihood), 0)

        param_net_input_from_prior = np.concatenate(
            (alpha_param_net_input, log_A_0[0]), 0
        )
        param_net_input_from_likelihood = np.concatenate((sumx, -np.array([N])), 0)
        param_net_input_full = np.concatenate(
            (param_net_input_from_prior, param_net_input_from_likelihood), 0
        )

        if param_net_input_type == "eta":
            param_net_input = param_net_input_full
        elif param_net_input_type == "prior":
            param_net_input = param_net_input_from_prior
        elif param_net_input_type == "likelihood":
            param_net_input = param_net_input_from_likelihood
        elif param_net_input_type == "data":
            assert x.shape[1] == 1 and N == 1
            param_net_input = x.T
        return eta, param_net_input

    def get_efn_dims(self, param_net_input_type="eta", give_hint=False):
        """Returns EFN component dimensionalities for the family.

		Args:
			param_net_input_type (str): specifies input to param network
				'eta':        give full eta to parameter network
				'prior':      part of eta that is prior-dependent
				'likelihood': part of eta that is likelihood-dependent
				'data':       the data itself
			give_hint (bool): Feed in prior covariance cholesky if true.

		Returns:
			D_Z (int): dimensionality of density network
			num_suff_stats: dimensionality of eta
			num_param_net_inputs: dimensionality of param net input
			num_T_z_inputs: dimensionality of suff stat comp input
		"""

        if give_hint:
            param_net_inputs_from_prior = self.num_prior_suff_stats + int(
                self.D * (self.D + 1) / 2
            )
        else:
            param_net_inputs_from_prior = self.num_prior_suff_stats

        if param_net_input_type == "eta":
            num_param_net_inputs = (
                param_net_inputs_from_prior + self.num_likelihood_suff_stats
            )
        elif param_net_input_type == "prior":
            num_param_net_inputs = param_net_inputs_from_prior
        elif param_net_input_type == "likelihood":
            num_param_net_inputs = self.num_likelihood_suff_stats
        elif param_net_input_type == "data":
            num_param_net_inputs = self.D

        return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_z_inputs


class LogGaussianCox(PosteriorFamily):
    """Log gaussian Cox family.

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		D_Z (int): dimensionality of the density network
		num_prior_suff_stats (int): number of suff stats that come from prior
		num_likelihood_suff_stats (int): " " from likelihood
		num_suff_stats (int): total number of suff stats
		num_T_z_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""

    def __init__(self, D, T=1, eta_dist=None, prior=[]):
        """truncated_normal_poisson family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""

        super().__init__(D, T, eta_dist)
        self.name = "LogGaussianCox"
        self.D_Z = D
        self.num_prior_suff_stats = int(D + D * (D + 1) / 2) + 1
        self.num_likelihood_suff_stats = D + 1
        self.num_suff_stats = self.num_prior_suff_stats + self.num_likelihood_suff_stats
        self.num_T_z_inputs = 0
        self.prior_family = MultivariateNormal(D, T)
        self.prior = prior
        self.data_num_resps = None
        self.train_set = None
        self.test_set = None

    def map_to_support(self, layers, num_theta_params):
        """Augment density network with bijective mapping to support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.

		"""
        support_layer = ShiftLayer(name="ShiftLayer", dim=self.D_Z)
        num_theta_params += count_layer_params(support_layer)
        layers.append(support_layer)
        return layers, num_theta_params

    def compute_suff_stats(self, Z, Z_by_layer, T_z_input):
        """Compute sufficient statistics of density network samples.

		Args:
			Z (tf.Tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_z_input (tf.Tensor): Param-dependent input.

		Returns:
			T_z (tf.Tensor): Sufficient statistics of samples.

		"""

        assert self.T == 1
        Z_shape = tf.shape(Z)
        K = Z_shape[0]
        M = Z_shape[1]
        T_z_prior = self.prior_family.compute_suff_stats(Z, Z_by_layer, T_z_input)
        const = -tf.ones((K, M, 1), tf.float64)
        z = Z[:, :, :, 0]
        sum_exp_z = tf.expand_dims(tf.reduce_sum(tf.exp(Z[:, :, :, 0]), 2), 2)
        T_z = tf.concat((T_z_prior, const, z, sum_exp_z), 2)
        return T_z

    def compute_log_base_measure(self, Z):
        """Compute log base measure of density network samples.

		Args:
			Z (tf.Tensor): Density network samples.

		Returns:
			log_h_z (tf.Tensor): Log base measure of samples.
			
		"""
        assert self.T == 1
        Z_shape = tf.shape(Z)
        K = Z_shape[0]
        M = Z_shape[1]
        log_h_z = tf.zeros((K, M), dtype=tf.float64)
        return log_h_z

    def load_data(self,):
        datadir = "data/responses/"
        num_monkeys = 3
        num_neurons = [83, 59, 105]
        num_oris = 12
        num_trials = 200
        N = sum(num_neurons * num_oris)
        X = np.zeros((N, self.D_Z, num_trials))
        ind = 0
        for i in range(num_monkeys):
            monkey = i + 1
            neurons = num_neurons[i]
            for j in range(neurons):
                neuron = j + 1
                for k in range(num_oris):
                    ori = k + 1
                    M = sio.loadmat(
                        datadir
                        + "spike_counts_monkey%d_neuron%d_ori%d.mat"
                        % (monkey, neuron, ori)
                    )
                    X[ind, :, :] = M["x"][:, : self.D_Z].T
                    resp_info = {"monkey": monkey, "neuron": neuron, "ori": ori}
                    assert ind == self.resp_info_to_ind(resp_info)
                    ind = ind + 1
        self.data = X
        self.data_num_resps = X.shape[0]
        return X

    def select_train_test_sets(self, num_test):
        if not (isinstance(num_test, int) and num_test >= 0):
            print("Number of test set samples must be a non-negative integer.")
            exit()
        elif num_test > self.data_num_resps:
            print(
                "Asked for %d samples in test set, but only %d total responses."
                % (num_test, self.data_num_resps)
            )
            exit()

        if num_test == 0:
            self.test_set = []
            self.train_set = range(self.data_num_resps)
        else:
            self.test_set = np.sort(
                np.random.choice(self.data_num_resps, num_test, False)
            ).tolist()
            inds = range(self.data_num_resps)
            test_set_inds = [i in self.test_set for i in inds]
            train_set_inds = [not i for i in test_set_inds]
            self.train_set = list(compress(inds, train_set_inds))

        return self.train_set, self.test_set

    def resp_info_to_ind(self, resp_info):
        monkey = resp_info["monkey"]
        neuron = resp_info["neuron"]
        ori = resp_info["ori"]
        num_neurons = [83, 59, 105]
        num_oris = 12
        ind = (
            sum(num_neurons[: (monkey - 1)]) * num_oris
            + (neuron - 1) * num_oris
            + (ori - 1)
        )
        return ind

    def draw_etas(
        self, K, param_net_input_type="eta", give_hint=False, train=True, resp_info=None
    ):
        """Samples K times from eta prior and sets up parameter network input.

		Args:
			K (int): Number of distributions.
			param_net_input_type (str): specifies input to param network
				'eta':        give full eta to parameter network
				'prior':      part of eta that is prior-dependent
				'likelihood': part of eta that is likelihood-dependent
				'data':       the data itself
			give_hint (bool): Feed in prior covariance cholesky if true.

		Returns:
			eta (np.array): K canonical parameters.
			param_net_inputs (np.array): K corresponding inputs to parameter network.
			T_z_input (np.array): K corresponding suff stat computation inputs.
			params (list): K corresponding mean parameterizations.
			
		"""
        datadir = "data/responses/"
        _, _, num_param_net_inputs, _ = self.get_efn_dims(
            param_net_input_type, give_hint
        )
        eta = np.zeros((K, self.num_suff_stats))
        param_net_inputs = np.zeros((K, num_param_net_inputs))
        T_z_input = np.zeros((K, self.num_T_z_inputs))
        Ts = 0.02
        mean_log_FR = -2.5892
        var_log_FR = 0.4424
        mu = mean_log_FR * np.ones((self.D_Z,))
        tau = 0.025
        Sigma = var_log_FR * get_GP_Sigma(tau, self.D_Z, Ts)
        if isinstance(self.prior, dict):
            N = self.prior["N"]
        else:
            N = 200
        params = []

        if K == 1 and (resp_info is not None):
            data_sets = [self.resp_info_to_ind(resp_info)]
        else:
            if train:
                data_set_inds = np.random.choice(len(self.train_set), K, False)
                data_sets = [self.train_set[data_set_inds[i]] for i in range(K)]
            else:
                data_set_inds = np.random.choice(len(self.test_set), K, False)
                data_sets = [self.test_set[data_set_inds[i]] for i in range(K)]
        for k in range(K):
            x = self.data[data_sets[k]]
            params_k = {
                "mu": mu,
                "Sigma": Sigma,
                "x": x,
                "N": N,
                "data_ind": data_sets[k],
            }
            params.append(params_k)
            eta[k, :], param_net_inputs[k, :] = self.mu_to_eta(
                params_k, param_net_input_type, give_hint
            )
            T_z_input[k, :] = self.mu_to_T_z_input(params_k)
        return eta, param_net_inputs, T_z_input, params

    def mu_to_eta(self, params, param_net_input_type="eta", give_hint=False):
        """Maps mean parameters (mu) of distribution to canonical parameters (eta).

		Args:
			params (dict): Mean parameters of distribution.
			param_net_input_type (str): specifies input to param network
				'eta':        give full eta to parameter network
				'prior':      part of eta that is prior-dependent
				'likelihood': part of eta that is likelihood-dependent
				'data':       the data itself
			give_hint (bool): Feed in prior covariance cholesky if true.

		Returns:
			eta (np.array): Canonical parameters.
			param_net_input (np.array): Corresponding input to parameter network.
			
		"""

        mu = params["mu"]
        Sigma = params["Sigma"]
        x = params["x"]
        N = params["N"]
        x = x[:, :N]

        alpha, alpha_param_net_input = self.prior_family.mu_to_eta(
            params, "eta", give_hint
        )
        mu = np.expand_dims(mu, 1)
        log_A_0 = 0.5 * (
            np.dot(mu.T, np.dot(np.linalg.inv(Sigma), mu))
            + np.log(np.linalg.det(Sigma))
        )
        sumx = np.sum(x, 1)

        eta_from_prior = np.concatenate((alpha, log_A_0[0]), 0)
        eta_from_likelihood = np.concatenate((sumx, -np.array([N])), 0)
        eta = np.concatenate((eta_from_prior, eta_from_likelihood), 0)

        param_net_input_from_prior = np.concatenate(
            (alpha_param_net_input, log_A_0[0]), 0
        )
        param_net_input_from_likelihood = np.concatenate((sumx, -np.array([N])), 0)
        param_net_input_full = np.concatenate(
            (param_net_input_from_prior, param_net_input_from_likelihood), 0
        )

        if param_net_input_type == "eta":
            param_net_input = param_net_input_full
        elif param_net_input_type == "prior":
            param_net_input = param_net_input_from_prior
        elif param_net_input_type == "likelihood":
            param_net_input = param_net_input_from_likelihood
        elif param_net_input_type == "data":
            assert x.shape[1] == 1 and N == 1
            param_net_input = x.T
        return eta, param_net_input

    def get_efn_dims(self, param_net_input_type="eta", give_hint=False):
        """Returns EFN component dimensionalities for the family.

		Args:
			param_net_input_type (str): Specifies input to param network.
				'eta':        Give full eta to parameter network.
				'prior':      Part of eta that is prior-dependent.
				'likelihood': Part of eta that is likelihood-dependent.
				'data':       The data itself.
			give_hint (bool): Feed in prior covariance cholesky if true.

		Returns:
			D_Z (int): Dimensionality of density network.
			num_suff_stats: Dimensionality of eta.
			num_param_net_inputs: Dimensionality of parameter network input.
			num_T_z_inputs: Dimensionality of suff stat computation input.

		"""

        if give_hint:
            param_net_inputs_from_prior = self.num_prior_suff_stats + int(
                self.D * (self.D + 1) / 2
            )
        else:
            param_net_inputs_from_prior = self.num_prior_suff_stats

        if param_net_input_type == "eta":
            num_param_net_inputs = (
                param_net_inputs_from_prior + self.num_likelihood_suff_stats
            )
        elif param_net_input_type == "prior":
            num_param_net_inputs = param_net_inputs_from_prior
        elif param_net_input_type == "likelihood":
            num_param_net_inputs = self.num_likelihood_suff_stats
        elif param_net_input_type == "data":
            num_param_net_inputs = self.D

        return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_z_inputs

    def default_eta_dist(self,):
        """Construct default eta prior."""
        return None


class SurrogateSD(Family):
    """Maximum entropy distribution with smoothness (S) and dim (D) constraints.

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		D_Z (int): dimensionality of the density network
		num_suff_stats (int): total number of suff stats
		num_T_z_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""

    def __init__(self, D, T=1):
        """multivariate_normal family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""
        self.name = "SurrogateSD"
        self.D = D
        self.T = T
        self.num_T_z_inputs = 0
        self.constant_base_measure = True
        self.has_log_p = True
        self.D_Z = D
        self.num_suff_stats = int(D + D * (D + 1) / 2) * T + D * int((T - 1) * T / 2)
        self.set_T_z_names()

    def set_T_z_names(self,):
        self.T_z_names = []
        self.T_z_names_tf = []
        self.T_z_group_names = []

        set_T_z_S_names(
            self.T_z_names, self.T_z_names_tf, self.T_z_group_names, self.D, self.T
        )
        set_T_z_D_names(
            self.T_z_names, self.T_z_names_tf, self.T_z_group_names, self.D, self.T
        )

        return None

    def compute_suff_stats(self, Z, Z_by_layer, T_z_input):
        """Compute sufficient statistics of density network samples.

		Args:
			Z (tf.Tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_z_input (tf.Tensor): Param-dependent input.

		Returns:
			T_z (tf.Tensor): Sufficient statistics of samples.

		"""
        T_z_S = compute_T_z_S(Z, self.D, self.T)
        T_z_D = compute_T_z_D(Z, self.D, self.T)
        # collect suff stats
        T_z = tf.concat((T_z_S, T_z_D), axis=2)

        return T_z

    def compute_log_base_measure(self, Z):
        """Compute log base measure of density network samples.

		Args:
			Z (tf.Tensor): Density network samples.

		Returns:
			log_h_z (tf.Tensor): Log base measure of samples.
			
		"""
        Z_shape = tf.shape(Z)
        K = Z_shape[0]
        M = Z_shape[1]
        log_h_z = tf.ones((K, M), dtype=tf.float64)
        return log_h_z

    def compute_mu(self, params):
        """Compute the mean parameterization (mu) given the mean parameters.

        Args:
			params (dict): Mean parameters of distributions.

		Returns:
			mu (np.array): The mean parameterization vector of the exponential family.

		"""
        mu_S = compute_mu_S(params, self.D, self.T)
        mu_D = compute_mu_D(params, self.D, self.T)
        mu = np.concatenate((mu_S, mu_D), 0)

        return mu

    def get_efn_dims(self, param_net_input_type="eta", give_hint=False):
        """Returns EFN component dimensionalities for the family.

		Args:
			param_net_input_type (str): Specifies input to param network.
				'eta':        Give full eta to parameter network.
			give_hint (bool): No hint implemented.

		Returns:
			D_Z (int): Dimensionality of density network.
			num_suff_stats: Dimensionality of eta.
			num_param_net_inputs: Dimensionality of parameter network input.
			num_T_z_inputs: Dimensionality of suff stat computation input.
			
		"""

        num_param_net_inputs = None
        return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_z_inputs

    def log_p_np(self, Z, params):
        """Computes log probability of Z given params.

		Args:
			Z (np.array): Density network samples.
			params (dict): Mean parameters of distribution.

		Returns:
			log_p (np.array): Ground truth probability of Z given params.

		"""
        eps = 1e-6
        K, M, D, T = Z.shape
        mu = params["mu_ME"]
        Sigma = params["Sigma_ME"]
        Sigma += eps * np.eye(D * T)
        dist = scipy.stats.multivariate_normal(mean=mu, cov=Sigma)
        Z_DT = np.reshape(np.transpose(Z, [0, 1, 3, 2]), [K, M, int(D * T)])
        log_p_z = dist.logpdf(Z_DT)
        return log_p_z


class SurrogateSED(Family):
    """Maximum entropy distribution with smoothness (S), 
	   endpoints (E) and dim (D) constraints.

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		D_Z (int): dimensionality of the density network
		num_suff_stats (int): total number of suff stats
		num_T_z_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""

    def __init__(self, D, T, Tps, T_Cs):
        """multivariate_normal family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""
        self.name = "SurrogateSED"
        self.D = D
        self.T = T  # total number of time points across all conditions
        self.num_T_z_inputs = 0
        self.constant_base_measure = True
        self.has_log_p = False
        self.D_Z = D

        mu_S_len = compute_mu_S_len(D, T_Cs)
        mu_D_len = (D + D * (D + 1) / 2) * T

        self.num_suff_stats = int(mu_S_len + mu_D_len)
        self.Tps = Tps
        self.T_Cs = T_Cs
        self.set_T_z_names()

    def set_T_z_names(self,):
        self.T_z_names = []
        self.T_z_names_tf = []
        self.T_z_group_names = []
        C = len(self.T_Cs)
        count = 0

        set_T_z_S_names_E(
            self.T_z_names, self.T_z_names_tf, self.T_z_group_names, self.D, self.T_Cs
        )
        set_T_z_D_names_E(
            self.T_z_names, self.T_z_names_tf, self.T_z_group_names, self.D, self.T_Cs
        )

        return None

    def compute_suff_stats(self, Z, Z_by_layer, T_z_input):
        """Compute sufficient statistics of density network samples.

		Args:
			Z (tf.Tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_z_input (tf.Tensor): Param-dependent input.

		Returns:
			T_z (tf.Tensor): Sufficient statistics of samples.

		"""
        T_z_S = compute_T_z_S_E(Z, self.D, self.T_Cs)
        Z_no_EP = self.remove_extra_endpoints_tf(Z)
        T_z_D = compute_T_z_D(Z_no_EP, self.D, self.T)
        # collect suff stats
        T_z = tf.concat((T_z_S, T_z_D), axis=2)
        return T_z

    def compute_log_base_measure(self, Z):
        """Compute log base measure of density network samples.

		Args:
			Z (tf.Tensor): Density network samples.

		Returns:
			log_h_z (tf.Tensor): Log base measure of samples.
			
		"""
        Z_shape = tf.shape(Z)
        K = Z_shape[0]
        M = Z_shape[1]
        log_h_z = tf.ones((K, M), dtype=tf.float64)
        return log_h_z

    def compute_mu(self, params):
        """Compute the mean parameterization (mu) given the mean parameters.

        Args:
			params (dict): Mean parameters of distributions.

		Returns:
			mu (np.array): The mean parameterization vector of the exponential family.

		"""
        mu_S = compute_mu_S_E(params, self.D, self.Tps, self.T_Cs)
        mu_D = compute_mu_D(params, self.D, self.T)

        mu = np.concatenate((mu_S, mu_D), 0)
        return mu

    def get_efn_dims(self, param_net_input_type="eta", give_hint=False):
        """Returns EFN component dimensionalities for the family.

		Args:
			param_net_input_type (str): Specifies input to param network.
				'eta':        Give full eta to parameter network.
			give_hint (bool): No hint implemented.

		Returns:
			D_Z (int): Dimensionality of density network.
			num_suff_stats: Dimensionality of eta.
			num_param_net_inputs: Dimensionality of parameter network input.
			num_T_z_inputs: Dimensionality of suff stat computation input.
			
		"""

        num_param_net_inputs = None
        return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_z_inputs

    def remove_extra_endpoints_tf(self, Z):
        Z_shape = tf.shape(Z)
        K = Z_shape[0]
        M = Z_shape[1]
        C = len(self.T_Cs)
        Zs = []
        t_ind = 0
        for i in range(C):
            T_C_i = self.T_Cs[i]
            if i == 0:
                Z_i = tf.slice(Z, [0, 0, 0, 0], [K, M, self.D, T_C_i])
            else:
                Z_i = tf.slice(Z, [0, 0, 0, t_ind + 1], [K, M, self.D, T_C_i - 2])
            t_ind = t_ind + T_C_i
            Zs.append(Z_i)
        Z_no_EP = tf.concat(Zs, 3)
        return Z_no_EP


class GPDirichlet(Family):
    """Maximum entropy distribution with smoothness (S) and dim (D) constraints.

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		D_Z (int): dimensionality of the density network
		num_suff_stats (int): total number of suff stats
		num_T_z_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""

    def __init__(self, D, T=1):
        """multivariate_normal family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""
        self.name = "GPDirichlet"
        self.D = D
        self.T = T
        self.D_Z = D - 1
        self.num_T_z_inputs = 0
        self.constant_base_measure = False
        self.has_log_p = False
        self.num_suff_stats = D * T + D * int((T - 1) * T / 2)
        self.set_T_z_names()

    def set_T_z_names(self,):
        self.T_z_names = []
        self.T_z_names_tf = []
        self.T_z_group_names = []

        set_T_z_S_names(
            self.T_z_names, self.T_z_names_tf, self.T_z_group_names, self.D, self.T
        )
        set_T_z_Dirich_names(
            self.T_z_names, self.T_z_names_tf, self.T_z_group_names, self.D, self.T
        )

        return None

    def map_to_support(self, layers, num_theta_params):
        """Augment density network with bijective mapping to support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.

		"""
        support_layer = SimplexBijectionLayer()
        num_theta_params += count_layer_params(support_layer)
        layers.append(support_layer)
        return layers, num_theta_params

    def compute_suff_stats(self, Z, Z_by_layer, T_z_input):
        """Compute sufficient statistics of density network samples.

		Args:
			Z (tf.Tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_z_input (tf.Tensor): Param-dependent input.

		Returns:
			T_z (tf.Tensor): Sufficient statistics of samples.

		"""
        T_z_S = compute_T_z_S(Z, self.D, self.T)
        T_z_D = compute_T_z_Dirich(Z, self.D, self.T)
        # collect suff stats
        T_z = tf.concat((T_z_S, T_z_D), axis=2)

        return T_z

    def compute_log_base_measure(self, Z):
        """Compute log base measure of density network samples.

		Args:
			Z (tf.Tensor): Density network samples.

		Returns:
			log_h_z (tf.Tensor): Log base measure of samples.
			
		"""
        Z_shape = tf.shape(Z)
        K = Z_shape[0]
        M = Z_shape[1]
        log_h_z = tf.ones((K, M), dtype=tf.float64)
        return log_h_z

    def compute_mu(self, params):
        """Compute the mean parameterization (mu) given the mean parameters.

        Args:
			params (dict): Mean parameters of distributions.

		Returns:
			mu (np.array): The mean parameterization vector of the exponential family.

		"""
        alpha = params["alpha"]
        alpha_0 = np.sum(alpha)

        mean = alpha / alpha_0
        var = np.multiply(alpha, alpha_0 - alpha) / (np.square(alpha_0) * (alpha_0 + 1))

        # compute (S) part of mu
        kernel = params["kernel"]
        ts = params["ts"]
        _T = ts.shape[0]
        autocov_dim = int(_T * (_T - 1) / 2)
        mu_S = np.zeros((int(self.D * autocov_dim),))
        autocovs = np.zeros((self.D, _T))
        if kernel == "SE":  # squared exponential
            taus = params["taus"]

        ind = 0
        ds = np.zeros((autocov_dim,))
        for t1 in range(_T):
            for t2 in range(t1 + 1, _T):
                ds[ind] = ts[t2] - ts[t1]
                ind += 1

        for i in range(self.D):
            if kernel == "SE":
                mu_S[(i * autocov_dim) : ((i + 1) * autocov_dim)] = var[i] * np.exp(
                    -np.square(ds) / (2 * np.square(taus[i]))
                ) + np.square(mean[i])

                # compute (D) part of mu
        phi_0 = psi(alpha_0)
        mu_alpha = psi(alpha) - phi_0
        mu_D = np.reshape(np.tile(np.expand_dims(mu_alpha, 1), [1, _T]), [self.D * _T])

        mu = np.concatenate((mu_S, mu_D), 0)

        return mu

    def get_efn_dims(self, param_net_input_type="eta", give_hint=False):
        """Returns EFN component dimensionalities for the family.

		Args:
			param_net_input_type (str): Specifies input to param network.
				'eta':        Give full eta to parameter network.
			give_hint
             (bool): No hint implemented.

		Returns:
			D_Z (int): Dimensionality of density network.
			num_suff_stats: Dimensionality of eta.
			num_param_net_inputs: Dimensionality of parameter network input.
			num_T_z_inputs: Dimensionality of suff stat computation input.
			
		"""

        num_param_net_inputs = None
        return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_z_inputs


class GPEDirichlet(Family):
    """Maximum entropy distribution with smoothness (GP), endpoints (EP) 
	   and expected log constraints (Dirichlet)

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		D_Z (int): dimensionality of the density network
		num_suff_stats (int): total number of suff stats
		num_T_z_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""

    def __init__(self, D, T, Tps, T_Cs):
        """multivariate_normal family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""
        self.name = "GPEDirichlet"
        self.D = D
        self.T = T
        self.num_T_z_inputs = 0
        self.constant_base_measure = False
        self.has_log_p = False
        self.D_Z = D - 1

        mu_S_len = compute_mu_S_len(T_Cs)
        mu_D_len = D * T

        self.num_suff_stats = int(mu_S_len + mu_D_len)
        self.Tps = Tps
        self.T_Cs = T_Cs
        self.set_T_z_names()

    def set_T_z_names(self,):
        self.T_z_names = []
        self.T_z_names_tf = []
        self.T_z_group_names = []

        set_T_z_S_names_E(
            self.T_z_names, self.T_z_names_tf, self.T_z_group_names, self.D, self.T_Cs
        )
        set_T_z_Dirich_names_E(
            self.T_z_names, self.T_z_names_tf, self.T_z_group_names, self.D, self.T_Cs
        )

        return None

    def map_to_support(self, layers, num_theta_params):
        """Augment density network with bijective mapping to support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.

		"""
        support_layer = SimplexBijectionLayer()
        num_theta_params += count_layer_params(support_layer)
        layers.append(support_layer)
        return layers, num_theta_params

    def compute_suff_stats(self, Z, Z_by_layer, T_z_input):
        """Compute sufficient statistics of density network samples.

		Args:
			Z (tf.Tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_z_input (tf.Tensor): Param-dependent input.

		Returns:
			T_z (tf.Tensor): Sufficient statistics of samples.

		"""
        Z_shape = tf.shape(Z)
        K = Z_shape[0]
        M = Z_shape[1]

        T_z_S = compute_T_z_S_E(Z, self.D, self.T_Cs)
        Z_no_EP = self.remove_extra_endpoints_tf(Z)
        T_z_D = compute_T_z_Dirich(Z_no_EP, self.D, self.T)

        # collect suff stats
        T_z = tf.concat((T_z_S, T_z_D), axis=2)

        return T_z

    def compute_log_base_measure(self, Z):
        """Compute log base measure of density network samples.

		Args:
			Z (tf.Tensor): Density network samples.

		Returns:
			log_h_z (tf.Tensor): Log base measure of samples.
			
		"""
        Z_shape = tf.shape(Z)
        K = Z_shape[0]
        M = Z_shape[1]
        log_h_z = tf.ones((K, M), dtype=tf.float64)
        return log_h_z

    def compute_mu(self, params):
        """Compute the mean parameterization (mu) given the mean parameters.

        Args:
			params (dict): Mean parameters of distributions.

		Returns:
			mu (np.array): The mean parameterization vector of the exponential family.

		"""
        kernel = params["kernel"]
        alpha = params["alpha"]

        alpha_0 = np.sum(alpha)

        mean = alpha / alpha_0
        var = np.multiply(alpha, alpha_0 - alpha) / (np.square(alpha_0) * (alpha_0 + 1))

        ts = params["ts"]
        C = len(self.T_Cs)
        max_Tp = max(self.Tps)
        mu_S_len = 0
        for i in range(C):
            mu_S_len += int(self.T_Cs[i] * (self.T_Cs[i] - 1) / 2)
            if i > 0:
                mu_S_len = mu_S_len - 1

        mu_S = np.zeros((self.D * mu_S_len,))
        if kernel == "SE":  # squared exponential
            taus = params["taus"]

        ind = 0
        for i in range(C):
            T_C_i = self.T_Cs[i]
            Tp = self.Tps[i]
            ts_i = np.concatenate([np.array([0.0]), ts, np.array([Tp])])
            for d in range(self.D):
                for t1 in range(T_C_i):
                    for t2 in range(t1 + 1, T_C_i):
                        if i > 0 and t1 == 0 and t2 == (T_C_i - 1):
                            continue
                        mu_S[ind] = var[d] * np.exp(
                            -np.square(ts_i[t2] - ts_i[t1]) / (2 * np.square(taus[d]))
                        ) + np.square(mean[d])
                        ind = ind + 1

                        # compute (D) part of mu
        phi_0 = psi(alpha_0)
        mu_alpha = psi(alpha) - phi_0
        T_no_EP = self.T
        mu_D = np.reshape(
            np.tile(np.expand_dims(mu_alpha, 1), [1, T_no_EP]), [self.D * T_no_EP]
        )

        mu = np.concatenate((mu_S, mu_D), 0)
        return mu

    def get_efn_dims(self, param_net_input_type="eta", give_hint=False):
        """Returns EFN component dimensionalities for the family.

		Args:
			param_net_input_type (str): Specifies input to param network.
				'eta':        Give full eta to parameter network.
			give_hint (bool): No hint implemented.

		Returns:
			D_Z (int): Dimensionality of density network.
			num_suff_stats: Dimensionality of eta.
			num_param_net_inputs: Dimensionality of parameter network input.
			num_T_z_inputs: Dimensionality of suff stat computation input.
			
		"""

        num_param_net_inputs = None
        return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_z_inputs

    def remove_extra_endpoints_tf(self, Z):
        Z_shape = tf.shape(Z)
        K = Z_shape[0]
        M = Z_shape[1]
        C = len(self.T_Cs)
        Zs = []
        t_ind = 0
        for i in range(C):
            T_C_i = self.T_Cs[i]
            if i == 0:
                Z_i = tf.slice(Z, [0, 0, 0, 0], [K, M, self.D, T_C_i])
            else:
                Z_i = tf.slice(Z, [0, 0, 0, t_ind + 1], [K, M, self.D, T_C_i - 2])
            t_ind = t_ind + T_C_i
            Zs.append(Z_i)
        Z_no_EP = tf.concat(Zs, 3)
        return Z_no_EP


def family_from_str(exp_fam_str):
    if exp_fam_str in ["MultivariateNormal", "normal", "multivariate_normal"]:
        return MultivariateNormal
    elif exp_fam_str in ["Dirichlet", "dirichlet"]:
        return Dirichlet
    elif exp_fam_str in ["InvWishart", "inv_wishart"]:
        return InvWishart
    elif exp_fam_str in ["HierarchicalDirichlet", "hierarchical_dirichlet", "dir_dir"]:
        return HierarchicalDirichlet
    elif exp_fam_str in ["DirichletMultinomial", "dirichlet_multinomial", "dir_mult"]:
        return DirichletMultinomial
    elif exp_fam_str in ["TruncatedNormalPoission", "truncated_normal_poisson", "tnp"]:
        return TruncatedNormalPoisson
    elif exp_fam_str in ["LogGaussianCox", "log_gaussian_cox", "lgc"]:
        return LogGaussianCox

    elif exp_fam_str in ["SD"]:
        return surrogateSD
    elif exp_fam_str in ["SED"]:
        return surrogateSED
    elif exp_fam_str in ["GPDirichlet"]:
        return GPDirichlet
    elif exp_fam_str in ["GPEDirichlet"]:
        return GPEDirichlet


def compute_mu_S_len(D, T_Cs):
    C = len(T_Cs)
    mu_S_len = 0.0
    for i in range(C):
        mu_S_len += int(T_Cs[i] * (T_Cs[i] - 1) / 2)
        if i > 0:
            mu_S_len = mu_S_len - 1
    mu_S_len = D * mu_S_len
    return int(mu_S_len)


def set_T_z_S_names(T_z_names, T_z_names_tf, T_z_group_names, D, T):
    # autocovariance by dimension E[x_cda x_cdb]
    for d in range(D):
        for t1 in range(T):
            for t2 in range(t1 + 1, T):
                T_z_names.append(
                    "$x_{%d,%d}$ $x_{%d,%d}$" % (d + 1, t1 + 1, d + 1, t2 + 1)
                )
                T_z_names_tf.append("x_%d,%d x_%d,%d" % (d + 1, t1 + 1, d + 1, t2 + 1))
                T_z_group_names.append(
                    "$x_{%d,t}$ $x_{%d,t+%d}$" % (d + 1, d + 1, int(abs(t2 - t1)))
                )
    return None


def set_T_z_D_names(T_z_names, T_z_names_tf, T_z_group_names, D, T):
    # mean E[x_dt]
    for i in range(D):
        for t in range(T):
            T_z_names.append("$x_{%d,%d}$" % (i + 1, t + 1))
            T_z_names_tf.append("x_%d,%d" % (i + 1, t + 1))
            T_z_group_names.append("$x_{%d,t}$" % (i + 1))

            # time-invariant covariance E[x_at x_bt]
    for i in range(D):
        for j in range(i, D):
            for t in range(T):
                T_z_names.append(
                    "$x_{%d,%d}$ $x_{%d,%d}$" % (i + 1, t + 1, j + 1, t + 1)
                )
                T_z_names_tf.append("x_%d,%d x_%d,%d" % (i + 1, t + 1, j + 1, t + 1))
                T_z_group_names.append("$x_{%d,t}$ $x_{%d,t}$" % (i + 1, j + 1))
    return None


def set_T_z_Dirich_names(T_z_names, T_z_names_tf, T_z_group_names, D, T):
    # time-invariant expected log constraints
    for i in range(D):
        for t in range(T):
            T_z_names.append("$\log x_{%d,%d}$" % (i + 1, t + 1))
            T_z_names_tf.append("\log x_%d,%d" % (i + 1, t + 1))
            T_z_group_names.append("$\log x_{%d,t}$" % (i + 1))
    return None


def set_T_z_S_names_E(T_z_names, T_z_names_tf, T_z_group_names, D, T_Cs):
    # mindful of endpoints across conditions
    # autocovariance by dimension E[x_cda x_cdb]
    C = len(T_Cs)
    for i in range(C):
        T_C_i = T_Cs[i]
        for d in range(D):
            for t1 in range(T_C_i):
                for t2 in range(t1 + 1, T_C_i):
                    if i > 0 and t1 == 0 and t2 == (T_C_i - 1):
                        continue
                    T_z_names.append(
                        "$x_{%d,%d,%d}$ $x_{%d,%d,%d}$"
                        % (i + 1, d + 1, t1 + 1, i + 1, d + 1, t2 + 1)
                    )
                    T_z_names_tf.append(
                        "x_%d,%d,%d x_%d,%d,%d"
                        % (i + 1, d + 1, t1 + 1, i + 1, d + 1, t2 + 1)
                    )
                    T_z_group_names.append(
                        "$x_{%d,%d,t}$ $x_{%d,%d,t+%d}$"
                        % (i + 1, d + 1, i + 1, d + 1, int(abs(t2 - t1)))
                    )
    return None


def set_T_z_D_names_E(T_z_names, T_z_names_tf, T_z_group_names, D, T_Cs):
    # mindful of endpoints across conditions
    C = len(T_Cs)
    # mean E[x_cdt]
    for d in range(D):
        for i in range(C):
            T_C_i = T_Cs[i]
            if i == 0:
                ts = range(T_C_i)
            else:
                ts = range(1, T_C_i - 1)
            for t in ts:
                T_z_names.append("$x_{%d,%d,%d}$" % (i + 1, d + 1, t + 1))
                T_z_names_tf.append("x_%d,%d,%d" % (i + 1, d + 1, t + 1))
                T_z_group_names.append("$x_{%d,%d,t}$" % (i + 1, d + 1))

                # time-invariant covariance E[x_cat x_cbt]
    for d1 in range(D):
        for d2 in range(d1, D):
            for i in range(C):
                T_C_i = T_Cs[i]
                if i == 0:
                    ts = range(T_C_i)
                else:
                    ts = range(1, T_C_i - 1)
                for t in ts:
                    T_z_names.append(
                        "$x_{%d,%d,%d}$ $x_{%d,%d,%d}$"
                        % (i + 1, d1 + 1, t + 1, i + 1, d2 + 1, t + 1)
                    )
                    T_z_names_tf.append(
                        "x_%d,%d,%d x_%d,%d,%d"
                        % (i + 1, d1 + 1, t + 1, i + 1, d2 + 1, t + 1)
                    )
                    T_z_group_names.append(
                        "$x_{%d,%d,t}$ $x_{%d,%d,t}$" % (i + 1, d1 + 1, i + 1, d2 + 1)
                    )
    return None


def set_T_z_Dirich_names_E(T_z_names, T_z_names_tf, T_z_group_names, D, T_Cs):
    # mindful of endpoints across conditions
    # time-invariant expected log constraints
    C = len(T_Cs)
    for d in range(D):
        for i in range(C):
            T_C_i = T_Cs[i]
            if i == 0:
                ts = range(T_C_i)
            else:
                ts = range(1, T_C_i - 1)
            for t in ts:
                T_z_names.append("$\log x_{%d,%d,%d}$" % (i + 1, d + 1, t + 1))
                T_z_names_tf.append("\log x_%d,%d,%d" % (i + 1, d + 1, t + 1))
                T_z_group_names.append("$\log x_{c=%d,%d,t}$" % (i + 1, d + 1))

    return None


def compute_T_z_S(Z, D, T):
    Z_shape = tf.shape(Z)
    K = Z_shape[0]
    M = Z_shape[1]
    # compute the (S) suff stats
    ones_mat = tf.ones((T, T))
    mask_T = tf.matrix_band_part(ones_mat, 0, -1) - tf.matrix_band_part(ones_mat, 0, 0)
    cov_con_mask_T = tf.cast(mask_T, dtype=tf.bool)
    ZZT_KMDTT = tf.matmul(tf.expand_dims(Z, 4), tf.expand_dims(Z, 3))
    T_z_S_KMDTcov = tf.transpose(
        tf.boolean_mask(tf.transpose(ZZT_KMDTT, [3, 4, 0, 1, 2]), cov_con_mask_T),
        [1, 2, 3, 0],
    )
    T_z_S = tf.reshape(T_z_S_KMDTcov, [K, M, D * tf.cast(T * (T - 1) / 2, tf.int32)])
    return T_z_S


def compute_T_z_Dirich(Z, D, T):
    Z_shape = tf.shape(Z)
    K = Z_shape[0]
    M = Z_shape[1]
    # compute the (D) suff stats
    T_z_D = tf.reshape(tf.log(Z), [K, M, D * T])
    return T_z_D


def compute_T_z_D(X, D, T):
    Z_shape = tf.shape(Z)
    K = Z_shape[0]
    M = Z_shape[1]
    # compute the (D) suff stats
    cov_con_mask_D = np.triu(np.ones((D, D), dtype=np.bool_), 0)
    T_z_mean = tf.reshape(Z, [K, M, D * T])

    Z_KMTD = tf.transpose(Z, [0, 1, 3, 2])
    # samps z D
    ZZT_KMTDD = tf.matmul(tf.expand_dims(Z_KMTD, 4), tf.expand_dims(Z_KMTD, 3))
    T_z_cov_KMDcovT = tf.transpose(
        tf.boolean_mask(tf.transpose(ZZT_KMTDD, [3, 4, 0, 1, 2]), cov_con_mask_D),
        [1, 2, 0, 3],
    )
    T_z_cov = tf.reshape(T_z_cov_KMDcovT, [K, M, int(D * (D + 1) / 2) * T])
    T_z_D = tf.concat((T_z_mean, T_z_cov), axis=2)
    return T_z_D


def compute_T_z_S_E(Z, D, T_Cs):
    # compute the (S) suff stats
    Z_shape = tf.shape(Z)
    K = Z_shape[0]
    M = Z_shape[1]
    C = len(T_Cs)
    t_ind = 0
    T_z_Ss = []
    for i in range(C):
        T_C_i = T_Cs[i]
        Z_Tci = tf.slice(Z, [0, 0, 0, t_ind], [K, M, D, T_C_i])
        t_ind = t_ind + T_C_i
        cov_con_mask_T = np.triu(np.ones((T_C_i, T_C_i), dtype=np.bool_), 1)
        ZZT_KMDTT = tf.matmul(tf.expand_dims(Z_Tci, 4), tf.expand_dims(Z_Tci, 3))
        T_z_S_KMDTcov = tf.transpose(
            tf.boolean_mask(tf.transpose(ZZT_KMDTT, [3, 4, 0, 1, 2]), cov_con_mask_T),
            [1, 2, 3, 0],
        )
        if i > 0:
            T_z_S_KMDTcov = tf.concat(
                (
                    T_z_S_KMDTcov[:, :, :, : (T_C_i - 2)],
                    T_z_S_KMDTcov[:, :, :, (T_C_i - 1) :],
                ),
                3,
            )
            T_z_S_i = tf.reshape(
                T_z_S_KMDTcov, [K, M, D * int(T_C_i * (T_C_i - 1) / 2 - 1)]
            )
            # remove repeated endpoint correlation
        else:
            T_z_S_i = tf.reshape(
                T_z_S_KMDTcov, [K, M, D * int(T_C_i * (T_C_i - 1) / 2)]
            )
        T_z_Ss.append(T_z_S_i)
    T_z_S = tf.concat(T_z_Ss, 2)
    return T_z_S


def compute_mu_S(params, D, T):
    kernel = params["kernel"]
    mu = params["mu_D"]
    Sigma = params["Sigma_D"]
    ts = params["ts"]
    autocov_dim = int(T * (T - 1) / 2)
    mu_S = np.zeros((int(D * autocov_dim),))
    autocovs = np.zeros((D, T))
    if kernel == "SE":  # squared exponential
        taus = params["taus"]

    ind = 0
    ds = np.zeros((autocov_dim,))
    for t1 in range(T):
        for t2 in range(t1 + 1, T):
            ds[ind] = ts[t2] - ts[t1]
            ind += 1

    for i in range(D):
        if kernel == "SE":
            mu_S[(i * autocov_dim) : ((i + 1) * autocov_dim)] = Sigma[i, i] * np.exp(
                -np.square(ds) / (2 * np.square(taus[i]))
            ) + np.square(mu[i])
    return mu_S


def compute_mu_S_E(params, D, Tps, T_Cs):
    def parse_ts(ts, i):
        _ts = []
        for i in range(i + 1):
            if ts[i].size > 0:
                _ts.append(ts[i])
        return _ts

    mu = params["mu_D"]
    Sigma = params["Sigma_D"]
    kernel = params["kernel"]
    ts = params["ts"]
    if kernel == "SE":  # squared exponential
        taus = params["taus"]

    mu_S_len = compute_mu_S_len(D, T_Cs)
    mu_S = np.zeros((mu_S_len,))

    ind = 0
    C = len(T_Cs)
    for i in range(C):
        T_C_i = T_Cs[i]
        Tp = Tps[i]
        ts_i = np.concatenate(parse_ts(ts, i))
        ts_i = np.concatenate([np.array([0.0]), ts_i, np.array([Tp])])
        for d in range(D):
            for t1 in range(T_C_i):
                for t2 in range(t1 + 1, T_C_i):
                    if i > 0 and t1 == 0 and t2 == (T_C_i - 1):
                        continue
                    mu_S[ind] = Sigma[d, d] * np.exp(
                        -np.square(ts_i[t2] - ts_i[t1]) / (2 * np.square(taus[d]))
                    ) + np.square(mu[d])
                    ind = ind + 1
    return mu_S


def compute_mu_D(params, D, T):
    mu = params["mu_D"]
    Sigma = params["Sigma_D"]
    # compute (D) part of mu
    mu_mu = np.reshape(np.tile(mu, [1, T]), [D * T])
    mu_Sigma = np.zeros((int(D * (D + 1) / 2)))
    ind = 0
    for i in range(D):
        for j in range(i, D):
            mu_Sigma[ind] = Sigma[i, j] + mu[i] * mu[j]
            ind += 1
    mu_Sigma = np.reshape(
        np.tile(np.expand_dims(mu_Sigma, 1), [1, T]), [int(D * (D + 1) / 2) * T]
    )

    mu_D = np.concatenate((mu_mu, mu_Sigma), 0)

    return mu_D
