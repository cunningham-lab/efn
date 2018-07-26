import tensorflow as tf
import numpy as np
from efn_util import count_layer_params, truncated_multivariate_normal_rvs, get_GP_Sigma, \
                     drawPoissonCounts
from flows import SimplexBijectionLayer, CholProdLayer, SoftPlusLayer, ShiftLayer
import scipy.stats
from scipy.special import gammaln, psi
import scipy.io as sio
from itertools import compress

def family_from_str(exp_fam_str):
	if (exp_fam_str in ['normal', 'multivariate_normal']):
		return multivariate_normal;
	elif (exp_fam_str in ['dirichlet']):
		return dirichlet;
	elif (exp_fam_str in ['inv_wishart']):
		return inv_wishart;
	elif (exp_fam_str in ['hierarchical_dirichlet', 'dir_dir']):
		return hierarchical_dirichlet;
	elif (exp_fam_str in ['dirichlet_multinomial', 'dir_mult']):
		return dirichlet_multinomial;
	elif (exp_fam_str in ['truncated_normal_poisson', 'prp_tn', 'tnp']):
		return truncated_normal_poisson;
	elif (exp_fam_str in ['log_gaussian_cox', 'lgc']):
		return log_gaussian_cox;

	elif (exp_fam_str in ['S_D', 'D_S']):
		return surrogate_S_D;
	elif (exp_fam_str in ['S_D_nodyn', 'D_S_nodyn']):
		return surrogate_S_D_nodyn;
	elif (exp_fam_str in ['S_D_C', 'S_C_D', 'D_S_C', 'D_C_S', 'C_S_D', 'C_D_S']):
		return surrogate_S_D_C;
	elif (exp_fam_str in ['GP_Dirichlet']):
		return GP_Dirichlet;
	elif (exp_fam_str in ['Fake_GP_Dirichlet']):
		return Fake_GP_Dirichlet;

def autocov_tf(X, tau_max, T):
    # need to finish this
    X_shape = tf.shape(X);
    K = X_shape[0];
    M = X_shape[1];
    D = X_shape[2];
    X_toep = [];
    for i in range(tau_max+1):
        X_toep.append(tf.concat((X[:,:,:,i:], tf.zeros((K,M,D,i), dtype=tf.float64)), 3));  
    X_toep = tf.transpose(tf.convert_to_tensor(X_toep), [1, 2, 3, 0, 4]);
    X_toep_1 = tf.expand_dims(X_toep[:,:,:,0,:], 4);

    num_samps = 1.0 / (T - tf.range(1, tau_max+1, dtype=tf.float64));
    num_samps_bcast = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(num_samps, 0), 0), 0), 4);
    
    autocov = tf.matmul(X_toep[:,:,:,1:,:], X_toep_1);
    print(autocov);
    autocov = tf.multiply(autocov, num_samps_bcast);
    print(autocov);
    return autocov;

class family:
	"""Base class for exponential families.
	
	Exponential families differ in their sufficient statistics, base measures,
	supports, ane therefore their natural parametrization.  Derivatives of this 
	class are useful tools for learning exponential family models in tensorflow.

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		num_T_x_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""

	def __init__(self, D, T=1):
		"""family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""

		self.D = D;
		self.T = T;
		self.realT = T;
		self.num_T_x_inputs = 0;
		self.constant_base_measure = True;
		self.has_log_p = False;

	def get_efn_dims(self, param_net_input_type='eta', give_hint=False):
		"""Returns EFN component dimensionalities for the family."""

	def map_to_support(self, layers, num_theta_params):
		"""Augment density network with bijective mapping to support."""
		return layers, num_theta_params;

	def compute_suff_stats(self, X, Z_by_layer, T_x_input):
		"""Compute sufficient statistics of density network samples."""
		raise NotImplementedError();

	def compute_mu(self, params):
		"""No comment yet."""
		raise NotImplementedError();

	def center_suff_stats_by_mu(self, T_x, mu):
		"""Center sufficient statistics by the mean parameters mu."""
		return T_x - tf.expand_dims(tf.expand_dims(mu, 0), 1);

	def compute_log_base_measure(self, X):
		"""Compute log base measure of density network samples."""
		raise NotImplementedError();

	def draw_etas(self, K, param_net_input_type='eta', give_hint=False):
		"""TODO I expect this function to change with prior specification ."""
		raise NotImplementedError();

	def mu_to_eta(self, params, param_net_input_type='eta', give_hint='False'):
		"""Maps mean parameters (mu) of distribution to canonical parameters (eta)."""
		raise NotImplementedError();

	def mu_to_T_x_input(self, params):
		"""Maps mean parameters (mu) of distribution to suff stat comp input.

		Args:
			params (dict): Mean parameters.

		Returns:
			T_x_input (np.array): Param-dependent input.
		"""

		T_x_input = np.array([]);
		return T_x_input;

	def log_p(self, X, params):
		"""Computes log probability of X given params.

		Args:
			X (tf.tensor): density network samples
			params (dict): Mean parameters.

		Returns:
			log_p (np.array): Ground truth probability of X for params.
		"""

		raise NotImplementedError();
		return None;

	def batch_diagnostics(self, K, sess, feed_dict, X, log_p_x, elbos, R2s, eta_draw_params, checkEntropy=False):
		"""Returns elbos, r^2s, and KL divergences of K distributions of family.

		Args:
			K (int): number of distributions
			sess (tf session): current tf session
			feed_dict (dict): contains Z0, eta, param_net_input, and T_x_input
			X (tf.tensor): density network samples
			log_p_x (tf.tensor): log probs of X
			elbos (tf.tensor): ELBO for each distribution
			R2s (tf.tensor): r^2 for each distribution
			eta_draw_params (list): contains mean parameters of each distribution
			check_entropy (bool): print model entropy relative to true entropy

		Returns:
			_elbos (np.array): approximate ELBO for each dist
			_R2s (np.array): approximate r^2s for each dist
			KLs (np.array): approximate KL divergence for each dist
		"""

		_X, _log_p_x, _elbos, _R2s = sess.run([X, log_p_x, elbos, R2s], feed_dict);
		KLs = [];
		for k in range(K):
			log_p_x_k = _log_p_x[k,:];
			X_k = _X[k, :, :, 0]; # TODO update this for time series
			params_k = eta_draw_params[k];
			KL_k = self.approx_KL(log_p_x_k, X_k, params_k);
			KLs.append(KL_k);
			if (checkEntropy):
				self.check_entropy(log_p_x_k, params_k);
		return _elbos, _R2s, KLs, _X;

	def approx_KL(self, log_Q, X, params):
		"""Approximate KL(Q || P)."""
		return np.nan;

	def approx_entropy(self, log_Q):
		"""Approximates entropy of the sampled distribution.

		Args:
			log_Q (np.array): log probability of Q

		Returns:
			H (np.float): approximate entropy of Q
		"""

		return np.mean(-log_Q);

	def true_entropy(self, params):
		"""Calculates true entropy of the distribution from mean parameters."""
		return np.nan;

	def check_entropy(self, log_Q, params):
		"""Prints entropy of approximate distribution relative to target distribution.

		Args:
			log_Q (np.array): log probability of Q.
			params (dict): Mean parameters of P.
		"""

		approxH = self.approx_entropy(log_Q);
		trueH = self.true_entropy(params)
		if (not np.isnan(trueH)):
			print('model entropy / true entropy');
			print('%.2E / %.2E' % (approxH, trueH));
		else:
			print('model entropy');
			print('%.2E' % approxH);
		return None;

class posterior_family(family):
	"""Base class for posterior-inference exponential families.
	
	When the likelihood of a bayesian model has exoponential family form and is
	closed under sampling, we can learn the posterior-inference exponential
	family.  See section A.2 of the efn code docs.

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		D_Z (int): dimensionality of the density network
		num_prior_suff_stats (int): number of suff stats that come from prior
		num_likelihood_suff_stats (int): " " from likelihood
		num_suff_stats (int): total number of suff stats
		num_T_x_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""


	def __init__(self, D, T=1):
		"""posterior family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""
		super().__init__(D,T);
		self.D_Z = None;
		self.num_prior_suff_stats = None;
		self.num_likelihood_suff_stats = None;
		self.num_suff_stats = None;

	def get_efn_dims(self, param_net_input_type='eta', give_hint=False):
		"""Returns EFN component dimensionalities for the family.

		Args:
			param_net_input_type (str): specifies input to param network
				'eta':        give full eta to parameter network
				'prior':      part of eta that is prior-dependent
				'likelihood': part of eta that is likelihood-dependent
				'data':       the data itself
			give_hint: (bool): No hint implemented.

		Returns:
			D_Z (int): dimensionality of density network
			num_suff_stats: dimensionality of eta
			num_param_net_inputs: dimensionality of param net input
			num_T_x_inputs: dimensionality of suff stat comp input
		"""
		if (give_hint):
			raise NotImplementedError();
		if (param_net_input_type == 'eta'):
			num_param_net_inputs = self.num_suff_stats;
		elif (param_net_input_type == 'prior'):
			num_param_net_inputs = self.num_prior_suff_stats;
		elif (param_net_input_type == 'likelihood'):
			num_param_net_inputs = self.num_likelihood_suff_stats;
		elif (param_net_input_type == 'data'):
			num_param_net_inputs = self.D;
		return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_x_inputs;

class multivariate_normal(family):
	"""Multivariate normal family.

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		D_Z (int): dimensionality of the density network
		num_suff_stats (int): total number of suff stats
		num_T_x_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""

	def __init__(self, D, T=1):
		"""multivariate_normal family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""
		super().__init__(D, T);
		self.name = 'normal';
		self.D_Z = D;
		self.num_suff_stats = int(D+D*(D+1)/2);
		self.has_log_p = True;

	def get_efn_dims(self, param_net_input_type='eta', give_hint=False):
		"""Returns EFN component dimensionalities for the family.

		Args:
			param_net_input_type (str): specifies input to param network
				'eta':        give full eta to parameter network
			give_hint: (bool): Feed in covariance cholesky if true.

		Returns:
			D_Z (int): dimensionality of density network
			num_suff_stats: dimensionality of eta
			num_param_net_inputs: dimensionality of param net input
			num_T_x_inputs: dimensionality of suff stat comp input
		"""

		if (not param_net_input_type == 'eta'):
			raise NotImplementedError();

		if (give_hint):
			num_param_net_inputs = int(self.D + self.D*(self.D+1));
		else:
			num_param_net_inputs = int(self.D + self.D*(self.D+1)/2);
		return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_x_inputs;

	def compute_suff_stats(self, X, Z_by_layer, T_x_input):
		"""Compute sufficient statistics of density network samples.

		Args:
			X (tf.tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_x_input (tf.tensor): Param-dependent input.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		cov_con_mask = np.triu(np.ones((self.D,self.D), dtype=np.bool_), 0);
		T_x_mean = tf.reduce_mean(X, 3);
		X_KMTD = tf.transpose(X, [0, 1, 3, 2]); # samps x D
		XXT_KMTDD = tf.matmul(tf.expand_dims(X_KMTD, 4), tf.expand_dims(X_KMTD, 3));
		T_x_cov_KMTDZ = tf.transpose(tf.boolean_mask(tf.transpose(XXT_KMTDD, [3,4,0,1,2]), cov_con_mask), [1, 2, 3, 0]);
		T_x_cov = tf.reduce_mean(T_x_cov_KMTDZ, 2);
		T_x = tf.concat((T_x_mean, T_x_cov), axis=2);
		return T_x;

	def compute_mu(self, params):
		mu = params['mu'];
		Sigma = params['Sigma'];
		mu_mu = mu;
		mu_Sigma = np.zeros((int(self.D*(self.D+1)/2)),);
		ind = 0;
		for i in range(self.D):
			for j in range(i,self.D):
				mu_Sigma[ind] = Sigma[i,j] + mu[i]*mu[j];
				ind += 1;

		mu = np.concatenate((mu_mu, mu_Sigma), 0);
		return mu;


	def compute_log_base_measure(self, X):
		"""Compute log base measure of density network samples.

		Args:
			X (tf.tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_x_input (tf.tensor): Param-dependent input.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""

		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		log_h_x = -(self.D/2)*np.log(2*np.pi)*tf.ones((K,M), dtype=tf.float64);
		return log_h_x;

	def draw_etas(self, K, param_net_input_type='eta', give_hint=False):
		_, _, num_param_net_inputs, _ = self.get_efn_dims(param_net_input_type, give_hint);
		eta = np.zeros((K, self.num_suff_stats));
		param_net_inputs = np.zeros((K, num_param_net_inputs));
		T_x_input = np.zeros((K, self.num_T_x_inputs));
		df_fac = 5;
		df = df_fac*self.D_Z;
		Sigma_dist = scipy.stats.invwishart(df=df, scale=df*np.eye(self.D_Z));
		params = [];
		for k in range(K):
			mu_k = np.random.multivariate_normal(np.zeros((self.D_Z,)), np.eye(self.D_Z));
			Sigma_k = Sigma_dist.rvs(1);
			params_k = {'mu':mu_k, 'Sigma':Sigma_k};
			params.append(params_k);
			eta[k,:], param_net_inputs[k,:] = self.mu_to_eta(params_k, param_net_input_type, give_hint);
			T_x_input[k,:] = self.mu_to_T_x_input(params_k);
		return eta, param_net_inputs, T_x_input, params;

	def mu_to_eta(self, params, param_net_input_type='eta', give_hint='False'):
		"""Maps mean parameters (mu) of distribution to canonical parameters (eta).

		Args:
			params (dict): Mean parameters
			param_net_input_type (str): specifies input to param network
				'eta':        give full eta to parameter network
			give_hint: (bool): Feed in covariance cholesky if true.

		Returns:
			eta (np.array): Canonical parameters.
			param_net_input: Corresponding input to parameter network.
		"""

		if (not param_net_input_type == 'eta'):
			raise NotImplementedError();
		mu = params['mu'];
		Sigma = params['Sigma'];
		cov_con_inds = np.triu_indices(self.D_Z, 0);
		upright_tri_inds = np.triu_indices(self.D_Z, 1);
		chol_inds = np.tril_indices(self.D_Z, 0);
		eta1 = np.float64(np.dot(np.linalg.inv(Sigma), np.expand_dims(mu, 1))).T;
		eta2 = np.float64(-np.linalg.inv(Sigma) / 2);
		# by using the minimal representation, we need to multiply eta by two
		# for the off diagonal elements
		eta2[upright_tri_inds] = 2*eta2[upright_tri_inds];
		eta2_minimal = eta2[cov_con_inds];
		eta = np.concatenate((eta1[0], eta2_minimal));

		if (give_hint):
			L = np.linalg.cholesky(Sigma);
			chol_minimal = L[chol_inds];
			param_net_input = np.concatenate((eta, chol_minimal));
		else:
			param_net_input = eta;
		return eta, param_net_input;

	def log_p(self, X, params):
		"""Computes log probability of X given params.

		Args:
			X (tf.tensor): density network samples
			params (dict): Mean parameters.

		Returns:
			log_p (np.array): Ground truth probability of X for params.
		"""

		mu = params['mu'];
		Sigma = params['Sigma'];
		dist = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=Sigma);
		#dist = scipy.stats.multivariate_normal(mean=mu, cov=Sigma);
		assert(self.T == 1);
		log_p_x = dist.log_prob(X[:,:,:,0]);
		return log_p_x;

	def log_p_np(self, X, params):
		"""Computes log probability of X given params.

		Args:
			X (tf.tensor): density network samples
			params (dict): Mean parameters.

		Returns:
			log_p (np.array): Ground truth probability of X for params.
		"""

		mu = params['mu'];
		Sigma = params['Sigma'];
		dist = scipy.stats.multivariate_normal(mean=mu, cov=Sigma);
		assert(self.T == 1);
		log_p_x = dist.logpdf(X);
		return log_p_x;

	def approx_KL(self, log_Q, X, params):
		"""Approximate KL(Q || P).

		Args:
			log_Q (np.array): log prob of density network samples.
			X (np.array): Density network samples.
			params (dict): Mean parameters of target distribution.

		Returns:
			KL (np.float): KL(Q || P)
		"""

		log_P = self.log_p_np(X, params);
		KL = np.mean(log_Q - log_P);
		return KL;

	def true_entropy(self, params):
		"""Calculates true entropy of the distribution from mean parameters.

		Args:
			params (dict): Mean parameters

		Returns:
			H_true (np.float): True (enough) distribution entropy.
		"""

		mu = params['mu'];
		Sigma = params['Sigma'];
		dist = scipy.stats.multivariate_normal(mean=mu, cov=Sigma);
		H_true = dist.entropy();
		return H_true;

class dirichlet(family):
	"""Dirichlet family.

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		D_Z (int): dimensionality of the density network
		num_suff_stats (int): total number of suff stats
		num_T_x_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""

	def __init__(self, D, T=1):
		"""dirichlet family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""
		super().__init__(D, T);
		self.name = 'dirichlet';
		self.D_Z = D-1;
		self.num_suff_stats = D;
		self.constant_base_measure = False;
		self.has_log_p = True;

	def get_efn_dims(self, param_net_input_type='eta', give_hint=False):
		"""Returns EFN component dimensionalities for the family.

		Args:
			param_net_input_type (str): specifies input to param network
				'eta':        give full eta to parameter network
			give_hint: (bool): No hint implemented.

		Returns:
			D_Z (int): dimensionality of density network
			num_suff_stats: dimensionality of eta
			num_param_net_inputs: dimensionality of param net input
			num_T_x_inputs: dimensionality of suff stat comp input
		"""

		if (give_hint or (not param_net_input_type == 'eta')):
			raise NotImplementedError();
		num_param_net_inputs = self.D;			
		return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_x_inputs;

	def map_to_support(self, layers, num_theta_params):
		"""Augment density network with bijective mapping to support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.
		"""
		support_layer = SimplexBijectionLayer();
		num_theta_params += count_layer_params(support_layer);
		layers.append(support_layer);
		return layers, num_theta_params;

	def compute_suff_stats(self, X, Z_by_layer, T_x_input):
		"""Compute sufficient statistics of density network samples.

		Args:
			X (tf.tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_x_input (tf.tensor): Param-dependent input.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""

		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		log_X = tf.log(X);
		T_x_log = tf.reduce_mean(log_X, 3);
		T_x = T_x_log;
		return T_x;

	def compute_mu(self, params):
		alpha = params['alpha'];
		alpha_0 = np.sum(alpha);
		phi_0 = psi(alpha_0);
		mu = psi(alpha) - phi_0;
		return mu;

	def compute_log_base_measure(self, X):
		"""Compute log base measure of density network samples.

		Args:
			X (tf.tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_x_input (tf.tensor): Param-dependent input.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""

		assert(self.T == 1);
		log_h_x = -tf.reduce_sum(tf.log(X), [2]);
		return log_h_x[:,:,0];

	def draw_etas(self, K, param_net_input_type='eta', give_hint=False):
		_, _, num_param_net_inputs, _ = self.get_efn_dims(param_net_input_type, give_hint);
		eta = np.zeros((K, self.num_suff_stats));
		param_net_inputs = np.zeros((K, num_param_net_inputs));
		T_x_input = np.zeros((K, self.num_T_x_inputs));
		params = [];
		for k in range(K):
			alpha_k = np.random.uniform(.5, 5, (self.D,));
			params_k = {'alpha':alpha_k};
			params.append(params_k);
			eta[k,:], param_net_inputs[k,:] = self.mu_to_eta(params_k, param_net_input_type, give_hint);
			T_x_input[k,:] = self.mu_to_T_x_input(params_k);
		return eta, param_net_inputs, T_x_input, params;

	def mu_to_eta(self, params, param_net_input_type='eta', give_hint=False):
		"""Maps mean parameters (mu) of distribution to canonical parameters (eta).

		Args:
			params (dict): Mean parameters
			param_net_input_type (str): specifies input to param network
				'eta':        give full eta to parameter network
			give_hint: (bool): No hint implemented.

		Returns:
			eta (np.array): Canonical parameters.
			param_net_input: Corresponding input to parameter network.
		"""

		if (give_hint or (not param_net_input_type == 'eta')):
			raise NotImplementedError();
		alpha = params['alpha'];
		eta = alpha;
		param_net_input = alpha;
		return eta, param_net_input;

	def log_p(self, X, params):
		"""Computes log probability of X given params.

		Args:
			X (tf.tensor): density network samples
			params (dict): Mean parameters.

		Returns:
			log_p (np.array): Ground truth probability of X for params.
		"""

		alpha = params['alpha'];
		dist = tf.contrib.distributions.Dirichlet(alpha)
		assert(self.T == 1);
		log_p_x = dist.log_prob(X[:,:,:,0]);
		return log_p_x;

	def log_p_np(self, X, params):
		"""Computes log probability of X given params.

		Args:
			X (tf.tensor): density network samples
			params (dict): Mean parameters.

		Returns:
			log_p (np.array): Ground truth probability of X for params.
		"""
		nonzero_simplex_eps = 1e-32;
		alpha = params['alpha'];
		dist = scipy.stats.dirichlet(np.float64(alpha));
		X = np.float64(X) + nonzero_simplex_eps;
		X = X / np.expand_dims(np.sum(X, 1), 1);
		log_p_x = dist.logpdf(X.T);
		return log_p_x;

	def approx_KL(self, log_Q, X, params):
		"""Approximate KL(Q || P).

		Args:
			log_Q (np.array): log prob of density network samples.
			X (np.array): Density network samples.
			params (dict): Mean parameters of target distribution.

		Returns:
			KL (np.float): KL(Q || P)
		"""

		log_P = self.log_p_np(X, params);
		KL = np.mean(log_Q - log_P);
		return KL;

	def true_entropy(self, params):
		"""Calculates true entropy of the distribution from mean parameters.

		Args:
			params (dict): Mean parameters

		Returns:
			H_true (np.float): True (enough) distribution entropy.
		"""

		alpha = params['alpha'];
		dist = scipy.stats.dirichlet(np.float64(alpha));
		H_true = dist.entropy();
		return H_true;

class inv_wishart(family):
	"""Inverse-Wishart family.

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		D_Z (int): dimensionality of the density network
		num_suff_stats (int): total number of suff stats
		num_T_x_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""

	def __init__(self, D, T=1):
		"""inv_wishart family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""

		super().__init__(D, T);
		self.name = 'inv_wishart';
		self.sqrtD = int(np.sqrt(D));
		self.D_Z = int(self.sqrtD*(self.sqrtD+1)/2)
		self.num_suff_stats = self.D_Z + 1;
		self.has_log_p = True;

	def get_efn_dims(self, param_net_input_type='eta', give_hint=False):
		"""Returns EFN component dimensionalities for the family.

		Args:
			param_net_input_type (str): specifies input to param network
				'eta':        give full eta to parameter network
			give_hint: (bool): Feed in inverse Psi cholesky if True.

		Returns:
			D_Z (int): dimensionality of density network
			num_suff_stats: dimensionality of eta
			num_param_net_inputs: dimensionality of param net input
			num_T_x_inputs: dimensionality of suff stat comp input
		"""

		if (not param_net_input_type == 'eta'):
			raise NotImplementedError();

		if (give_hint):
			num_param_net_inputs = 2*self.D_Z + 1;
		else:
			num_param_net_inputs = self.num_suff_stats;
			
		return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_x_inputs;

	def map_to_support(self, layers, num_theta_params):
		"""Augment density network with bijective mapping to support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.
		"""
		support_layer = CholProdLayer();
		num_theta_params += count_layer_params(support_layer);
		layers.append(support_layer);
		return layers, num_theta_params;

	def compute_suff_stats(self, X, Z_by_layer, T_x_input):
		"""Compute sufficient statistics of density network samples.

		Args:
			X (tf.tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_x_input (tf.tensor): Param-dependent input.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""

		assert(self.T == 1);
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		cov_con_mask = np.triu(np.ones((self.sqrtD,self.sqrtD), dtype=np.bool_), 0);
		X = X[:,:,:,0]; # update for T > 1
		X_KMDsqrtDsqrtD = tf.reshape(X, (K,M,self.sqrtD,self.sqrtD));
		X_inv = tf.matrix_inverse(X_KMDsqrtDsqrtD);
		T_x_inv = tf.transpose(tf.boolean_mask(tf.transpose(X_inv, [2,3,0,1]), cov_con_mask), [1, 2, 0]);
		# We already have the Chol factor from earlier in the graph
		zchol = Z_by_layer[-2];
		zchol_KMD_Z = zchol[:,:,:,0]; # generalize this for more time points
		L = tf.contrib.distributions.fill_triangular(zchol_KMD_Z);
		L_pos_diag = tf.contrib.distributions.matrix_diag_transform(L, tf.exp)
		L_pos_diag_els = tf.matrix_diag_part(L_pos_diag);
		T_x_log_det = 2*tf.reduce_sum(tf.log(L_pos_diag_els), 2);
		T_x_log_det = tf.expand_dims(T_x_log_det, 2);
		T_x = tf.concat((T_x_inv, T_x_log_det), axis=2);
		return T_x;

	def compute_log_base_measure(self, X):
		"""Compute log base measure of density network samples.

		Args:
			X (tf.tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_x_input (tf.tensor): Param-dependent input.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""

		assert(self.T == 1);
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		log_h_x = tf.zeros((K,M), dtype=tf.float64);
		return log_h_x;

	def draw_etas(self, K, param_net_input_type='eta', give_hint=False):
		_, _, num_param_net_inputs, _ = self.get_efn_dims(param_net_input_type, give_hint);
		eta = np.zeros((K, self.num_suff_stats));
		param_net_inputs = np.zeros((K, num_param_net_inputs));
		T_x_input = np.zeros((K, self.num_T_x_inputs));

		df_fac = 100;
		df = df_fac*self.sqrtD;
		Psi_dist = scipy.stats.invwishart(df=df, scale=df*np.eye(self.sqrtD));
		params = [];
		for k in range(K):
			Psi_k = Psi_dist.rvs(1);
			m_k = np.random.randint(2*self.sqrtD,3*self.sqrtD+1)
			params_k = {'Psi':Psi_k, 'm':m_k};
			params.append(params_k);
			eta[k,:], param_net_inputs[k,:] = self.mu_to_eta(params_k, param_net_input_type, give_hint);
			T_x_input[k,:] = self.mu_to_T_x_input(params_k);
		return eta, param_net_inputs, T_x_input, params;

	def mu_to_eta(self, params, param_net_input_type='eta', give_hint=False):
		"""Maps mean parameters (mu) of distribution to canonical parameters (eta).

		Args:
			params (dict): Mean parameters
			param_net_input_type (str): specifies input to param network
				'eta':        give full eta to parameter network
			give_hint: (bool): Feed in inverse Psi cholesky if True.

		Returns:
			eta (np.array): Canonical parameters.
			param_net_input: Corresponding input to parameter network.
		"""

		if (not param_net_input_type == 'eta'):
			raise NotImplementedError();
		Psi = params['Psi'];
		m = params['m'];
		cov_con_inds = np.triu_indices(self.sqrtD, 0);
		upright_tri_inds = np.triu_indices(self.sqrtD, 1);
		eta1 = -Psi/2.0;
		eta1[upright_tri_inds] = 2*eta1[upright_tri_inds];
		eta1_minimal = eta1[cov_con_inds];
		eta2 = np.array([-(m+self.sqrtD+1)/2.0]);
		eta = np.concatenate((eta1_minimal, eta2));

		if (give_hint):
			Psi_inv = np.linalg.inv(Psi);
			Psi_inv_minimal = Psi_inv[cov_con_inds];
			param_net_input = np.concatenate((eta, Psi_inv_minimal));
		else:
			param_net_input = eta;
		return eta, param_net_input;

	def log_p(self, X, params, ):
		"""Computes log probability of X given params.

		Args:
			X (tf.tensor): density network samples
			params (dict): Mean parameters.

		Returns:
			log_p (np.array): Ground truth probability of X for params.
		"""

		batch_size = X.shape[0];
		Psi = params['Psi'];
		m = params['m'];
		X = np.reshape(X, [batch_size, self.sqrtD, self.sqrtD]);
		log_p_x = scipy.stats.invwishart.logpdf(np.transpose(X, [1,2,0]), m, Psi);
		return log_p_x;

	def approx_KL(self, log_Q, X, params):
		"""Approximate KL(Q || P).

		Args:
			log_Q (np.array): log prob of density network samples.
			X (np.array): Density network samples.
			params (dict): Mean parameters of target distribution.

		Returns:
			KL (np.float): KL(Q || P)
		"""

		
		log_P = self.log_p(X, params);
		KL = np.mean(log_Q - log_P);
		return KL;

class hierarchical_dirichlet(posterior_family):
	"""Hierarchical Dirichlet family.

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		D_Z (int): dimensionality of the density network
		num_prior_suff_stats (int): number of suff stats that come from prior
		num_likelihood_suff_stats (int): " " from likelihood
		num_suff_stats (int): total number of suff stats
		num_T_x_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""

	def __init__(self, D, T=1):
		"""hierarchical_dirichlet family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""

		super().__init__(D, T);
		self.name = 'hierarchical_dirichlet';
		self.D_Z = D-1;
		self.num_prior_suff_stats = D + 1;
		self.num_likelihood_suff_stats = D + 1;
		self.num_suff_stats = self.num_prior_suff_stats + self.num_likelihood_suff_stats;
		self.num_T_x_inputs = 1;

	def map_to_support(self, layers, num_theta_params):
		"""Augment density network with bijective mapping to support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.
		"""
		support_layer = SimplexBijectionLayer();
		num_theta_params += count_layer_params(support_layer);
		layers.append(support_layer);
		return layers, num_theta_params;

	def compute_suff_stats(self, X, Z_by_layer, T_x_input):
		"""Compute sufficient statistics of density network samples.

		Args:
			X (tf.tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_x_input (tf.tensor): Param-dependent input.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""

		assert(self.T == 1);
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		logz = tf.log(X[:,:,:,0]);
		const = -tf.ones((K,M,1), tf.float64);
		beta = tf.expand_dims(T_x_input, 1);
		betaz = tf.multiply(beta, X[:,:,:,0]);
		log_gamma_beta_z = tf.lgamma(betaz);
		log_gamma_beta = tf.lgamma(beta); # log(gamma(beta*sum(x_i))) = log(gamma(beta))
		log_Beta_beta_z = tf.expand_dims(tf.reduce_sum(log_gamma_beta_z, 2), 2) - log_gamma_beta;
		T_x = tf.concat((logz, const, betaz, log_Beta_beta_z), 2);
		return T_x;

	def compute_log_base_measure(self, X):
		"""Compute log base measure of density network samples.

		Args:
			X (tf.tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_x_input (tf.tensor): Param-dependent input.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""

		assert(self.T == 1);
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		log_h_x = tf.zeros((K,M), dtype=tf.float64);
		return log_h_x;

	def draw_etas(self, K, param_net_input_type='eta', give_hint=False):
		_, _, num_param_net_inputs, _ = self.get_efn_dims(param_net_input_type, give_hint);
		eta = np.zeros((K, self.num_suff_stats));
		param_net_inputs = np.zeros((K, num_param_net_inputs));
		T_x_input = np.zeros((K, self.num_T_x_inputs));
		Nmean = 5;
		x_eps = 1e-16;
		params = [];
		for k in range(K):
			alpha_0_k = np.random.uniform(1.0, 10.0, (self.D,));
			beta_k = np.random.uniform(self.D, 2*self.D);
			N = 1;
			#N = np.random.poisson(Nmean);
			dist1 = scipy.stats.dirichlet(alpha_0_k);
			z = dist1.rvs(1);
			dist2 = scipy.stats.dirichlet(beta_k*z[0]);
			x = dist2.rvs(N).T;
			x = (x+x_eps);
			x = x / np.expand_dims(np.sum(x, 0), 0);
			params_k = {'alpha_0':alpha_0_k, 'beta':beta_k, 'x':x, 'z':z, 'N':N};
			params.append(params_k);
			eta[k,:], param_net_inputs[k,:] = self.mu_to_eta(params_k, param_net_input_type, False);
			T_x_input[k,:] = self.mu_to_T_x_input(params_k);
		return eta, param_net_inputs, T_x_input, params;

	def mu_to_eta(self, params, param_net_input_type='eta', give_hint=False):
		"""Maps mean parameters (mu) of distribution to canonical parameters (eta).

		Args:
			params (dict): Mean parameters
			param_net_input_type (str): specifies input to param network
				'eta':        give full eta to parameter network
			give_hint: (bool): No hint implemented.

		Returns:
			eta (np.array): Canonical parameters.
			param_net_input: Corresponding input to parameter network.
		"""

		if (give_hint):
			raise NotImplementedError();
		alpha_0 = params['alpha_0'];
		x = params['x'];
		N = params['N'];
		assert(N == x.shape[1]);

		log_Beta_alpha_0 = np.array([np.sum(gammaln(alpha_0)) - gammaln(np.sum(alpha_0))]);
		sumlogx = np.sum(np.log(x), 1);

		eta_from_prior = np.concatenate((alpha_0-1.0, log_Beta_alpha_0), 0);
		eta_from_likelihood = np.concatenate((sumlogx, -np.array([N])), 0);
		eta = np.concatenate((eta_from_prior, eta_from_likelihood), 0);

		if (param_net_input_type == 'eta'):
			param_net_input = eta;
		elif (param_net_input_type == 'prior'):
			param_net_input = eta_from_prior;
		elif (param_net_input_type == 'likelihood'):
			param_net_input = eta_from_likelihood;
		elif (param_net_input_type == 'data'):
			assert(x.shape[1] == 1 and N == 1);
			param_net_input = x.T;
		return eta, param_net_input;

	def mu_to_T_x_input(self, params):
		"""Maps mean parameters (mu) of distribution to suff stat comp input.

		Args:
			params (dict): Mean parameters.

		Returns:
			T_x_input (np.array): Param-dependent input.
		"""

		beta = params['beta'];
		T_x_input = np.array([beta]);
		return T_x_input;

class dirichlet_multinomial(posterior_family):
	"""Dirichlet-multinomial family.

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		D_Z (int): dimensionality of the density network
		num_prior_suff_stats (int): number of suff stats that come from prior
		num_likelihood_suff_stats (int): " " from likelihood
		num_suff_stats (int): total number of suff stats
		num_T_x_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""

	def __init__(self, D, T=1):
		"""dirichlet_multinomial family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""

		super().__init__(D, T);
		self.name = 'dirichlet_multinomial';
		self.D_Z = D-1;
		self.num_prior_suff_stats = D + 1;
		self.num_likelihood_suff_stats = D + 1;
		self.num_suff_stats = self.num_prior_suff_stats + self.num_likelihood_suff_stats;
		self.num_T_x_inputs = 0;

	def map_to_support(self, layers, num_theta_params):
		"""Augment density network with bijective mapping to support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.
		"""
		support_layer = SimplexBijectionLayer();
		num_theta_params += count_layer_params(support_layer);
		layers.append(support_layer);
		return layers, num_theta_params;

	def compute_suff_stats(self, X, Z_by_layer, T_x_input):
		"""Compute sufficient statistics of density network samples.

		Args:
			X (tf.tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_x_input (tf.tensor): Param-dependent input.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""

		assert(self.T == 1);
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		logz = tf.log(X[:,:,:,0]);
		const = -tf.ones((K,M,1), tf.float64);
		zeros = -tf.zeros((K,M,1), tf.float64);
		T_x = tf.concat((logz, const, logz, zeros), 2);
		return T_x;

	def compute_log_base_measure(self, X):
		"""Compute log base measure of density network samples.

		Args:
			X (tf.tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_x_input (tf.tensor): Param-dependent input.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""

		assert(self.T == 1);
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		log_h_x = tf.zeros((K,M), dtype=tf.float64);
		return log_h_x;

	def draw_etas(self, K, param_net_input_type='eta', give_hint=False):
		_, _, num_param_net_inputs, _ = self.get_efn_dims(param_net_input_type, give_hint);
		eta = np.zeros((K, self.num_suff_stats));
		param_net_inputs = np.zeros((K, num_param_net_inputs));
		T_x_input = np.zeros((K, self.num_T_x_inputs));
		N = 1
		x_eps = 1e-16;
		params = [];
		for k in range(K):
			alpha_0_k = np.random.uniform(1.0, 10.0, (self.D,));
			dist1 = scipy.stats.dirichlet(alpha_0_k);
			z = dist1.rvs(1);
			dist2 = scipy.stats.dirichlet(z[0]);
			x = dist2.rvs(N).T;
			x = (x+x_eps);
			x = x / np.expand_dims(np.sum(x, 0), 0);
			params_k = {'alpha_0':alpha_0_k, 'x':x, 'z':z, 'N':N};
			params.append(params_k);
			eta[k,:], param_net_inputs[k,:] = self.mu_to_eta(params_k, param_net_input_type, False);
			T_x_input[k,:] = self.mu_to_T_x_input(params_k);
		return eta, param_net_inputs, T_x_input, params;

	def mu_to_eta(self, params, param_net_input_type='eta', give_hint=False):
		"""Maps mean parameters (mu) of distribution to canonical parameters (eta).

		Args:
			params (dict): Mean parameters
			param_net_input_type (str): specifies input to param network
				'eta':        give full eta to parameter network
			give_hint: (bool): No hint implemented.

		Returns:
			eta (np.array): Canonical parameters.
			param_net_input: Corresponding input to parameter network.
		"""

		if (give_hint):
			raise NotImplementedError();
		alpha_0 = params['alpha_0'];
		x = params['x'];
		N = params['N'];
		assert(N == x.shape[1]);

		log_Beta_alpha_0 = np.array([np.sum(gammaln(alpha_0)) - gammaln(np.sum(alpha_0))]);

		eta_from_prior = np.concatenate((alpha_0-1.0, log_Beta_alpha_0), 0);
		eta_from_likelihood = np.concatenate((x[:,0], -np.array([N])), 0);
		eta = np.concatenate((eta_from_prior, eta_from_likelihood), 0);

		if (param_net_input_type == 'eta'):
			param_net_input = eta;
		elif (param_net_input_type == 'prior'):
			param_net_input = eta_from_prior;
		elif (param_net_input_type == 'likelihood'):
			param_net_input = eta_from_likelihood;
		elif (param_net_input_type == 'data'):
			assert(x.shape[1] == 1 and N == 1);
			param_net_input = x.T;
		return eta, param_net_input;

class truncated_normal_poisson(posterior_family):
	"""Truncated normal Poisson family.

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		D_Z (int): dimensionality of the density network
		num_prior_suff_stats (int): number of suff stats that come from prior
		num_likelihood_suff_stats (int): " " from likelihood
		num_suff_stats (int): total number of suff stats
		num_T_x_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""

	def __init__(self, D, T=1):
		"""truncated_normal_poisson family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""

		super().__init__(D, T);
		self.name = 'truncated_normal_poisson';
		self.D_Z = D;
		self.num_prior_suff_stats = int(D+D*(D+1)/2) + 1;
		self.num_likelihood_suff_stats = D + 1;
		self.num_suff_stats = self.num_prior_suff_stats + self.num_likelihood_suff_stats;
		self.num_T_x_inputs = 0;
		self.prior_family = multivariate_normal(D, T);

	def get_efn_dims(self, param_net_input_type='eta', give_hint=False):
		"""Returns EFN component dimensionalities for the family.

		Args:
			param_net_input_type (str): specifies input to param network
				'eta':        give full eta to parameter network
				'prior':      part of eta that is prior-dependent
				'likelihood': part of eta that is likelihood-dependent
				'data':       the data itself
			give_hint: (bool): Feed in prior covariance cholesky if true.

		Returns:
			D_Z (int): dimensionality of density network
			num_suff_stats: dimensionality of eta
			num_param_net_inputs: dimensionality of param net input
			num_T_x_inputs: dimensionality of suff stat comp input
		"""

		if (give_hint):
			param_net_inputs_from_prior = self.num_prior_suff_stats + int(self.D*(self.D+1)/2);
		else:
			param_net_inputs_from_prior = self.num_prior_suff_stats;

		if (param_net_input_type == 'eta'):
			num_param_net_inputs = param_net_inputs_from_prior \
			                       + self.num_likelihood_suff_stats;
		elif (param_net_input_type == 'prior'):
			num_param_net_inputs = param_net_inputs_from_prior;
		elif (param_net_input_type == 'likelihood'):
			num_param_net_inputs = self.num_likelihood_suff_stats;
		elif (param_net_input_type == 'data'):
			num_param_net_inputs = self.D;

		return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_x_inputs;

	def map_to_support(self, layers, num_theta_params):
		"""Augment density network with bijective mapping to support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.
		"""
		support_layer = SoftPlusLayer();
		num_theta_params += count_layer_params(support_layer);
		layers.append(support_layer);
		return layers, num_theta_params;

	def compute_suff_stats(self, X, Z_by_layer, T_x_input):
		"""Compute sufficient statistics of density network samples.

		Args:
			X (tf.tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_x_input (tf.tensor): Param-dependent input.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""

		assert(self.T == 1);
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		T_x_prior = self.prior_family.compute_suff_stats(X, Z_by_layer, T_x_input);
		const = -tf.ones((K,M,1), tf.float64);
		logz = tf.log(X[:,:,:,0]);
		sumz = tf.expand_dims(tf.reduce_sum(X[:,:,:,0], 2), 2);
		T_x = tf.concat((T_x_prior, const, logz, sumz), 2);
		return T_x;

	def compute_log_base_measure(self, X):
		"""Compute log base measure of density network samples.

		Args:
			X (tf.tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_x_input (tf.tensor): Param-dependent input.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""

		assert(self.T == 1);
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		log_h_x = tf.zeros((K,M), dtype=tf.float64);
		return log_h_x;


	def draw_etas(self, K, param_net_input_type='eta', give_hint=False):
		_, _, num_param_net_inputs, _ = self.get_efn_dims(param_net_input_type, give_hint);
		eta = np.zeros((K, self.num_suff_stats));
		param_net_inputs = np.zeros((K, num_param_net_inputs));
		T_x_input = np.zeros((K, self.num_T_x_inputs));
		nneurons = 83;
		noris = 12;
		Ts = .02;
		mean_FR = 0.1169;
		var_FR = 0.0079;
		mu = mean_FR*np.ones((self.D_Z,));
		tau = .025;
		#Sigma = var_FR*np.eye(self.D_Z);
		Sigma = var_FR*get_GP_Sigma(tau, self.D_Z, Ts)
		params = [];
		data_sets = np.random.choice(nneurons*noris, K, False);
		for k in range(K):
			neuron = (data_sets[k] // noris) + 1;
			ori = np.mod(data_sets[k], noris) + 1;
			M = sio.loadmat(datadir + 'spike_counts_neuron%d_ori%d.mat' % (neuron, ori));
			x = M['x'][:,:self.D_Z].T;
			N = x.shape[1];
			params_k = {'mu':mu, 'Sigma':Sigma, 'x':x, 'N':N, 'monkey':1, 'neuron':neuron, 'ori':ori};
			params.append(params_k);
			eta[k,:], param_net_inputs[k,:] = self.mu_to_eta(params_k, param_net_input_type, give_hint);
			T_x_input[k,:] = self.mu_to_T_x_input(params_k);
		return eta, param_net_inputs, T_x_input, params;

	def mu_to_eta(self, params, param_net_input_type='eta', give_hint=False):
		"""Maps mean parameters (mu) of distribution to canonical parameters (eta).

		Args:
			params (dict): Mean parameters
			param_net_input_type (str): specifies input to param network
				'eta':        give full eta to parameter network
			give_hint: (bool): Feed in prior covariance cholesky if true.
		Returns:
			eta (np.array): Canonical parameters.
			param_net_input: Corresponding input to parameter network.
		"""
		
		mu = params['mu'];
		Sigma = params['Sigma'];
		x = params['x'];
		N = params['N'];
		assert(N == x.shape[1]);

		alpha, alpha_param_net_input = self.prior_family.mu_to_eta(params, param_net_input_type, give_hint);
		mu = np.expand_dims(mu, 1);
		log_A_0 = 0.5*(np.dot(mu.T, np.dot(np.linalg.inv(Sigma), mu)) + np.log(np.linalg.det(Sigma)));
		sumx = np.sum(x, 1);

		eta_from_prior = np.concatenate((alpha, log_A_0[0]), 0);
		eta_from_likelihood = np.concatenate((sumx, -np.array([N])), 0);
		eta = np.concatenate((eta_from_prior, eta_from_likelihood), 0);

		param_net_input_from_prior = np.concatenate((alpha_param_net_input, log_A_0[0]), 0);
		param_net_input_from_likelihood = np.concatenate((sumx, -np.array([N])), 0);
		param_net_input_full = np.concatenate((param_net_input_from_prior, param_net_input_from_likelihood), 0);

		if (param_net_input_type == 'eta'):
			param_net_input = param_net_input_full;
		elif (param_net_input_type == 'prior'):
			param_net_input = param_net_input_from_prior;
		elif (param_net_input_type == 'likelihood'):
			param_net_input = param_net_input_from_likelihood;
		elif (param_net_input_type == 'data'):
			assert(x.shape[1] == 1 and N == 1);
			param_net_input = x.T;
		return eta, param_net_input;

class log_gaussian_cox(posterior_family):
	"""Log gaussian Cox family.

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		D_Z (int): dimensionality of the density network
		num_prior_suff_stats (int): number of suff stats that come from prior
		num_likelihood_suff_stats (int): " " from likelihood
		num_suff_stats (int): total number of suff stats
		num_T_x_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""

	def __init__(self, D, T=1, prior=[]):
		"""truncated_normal_poisson family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""

		super().__init__(D, T);
		self.name = 'log_gaussian_cox';
		self.D_Z = D;
		self.num_prior_suff_stats = int(D+D*(D+1)/2) + 1;
		self.num_likelihood_suff_stats = D + 1;
		self.num_suff_stats = self.num_prior_suff_stats + self.num_likelihood_suff_stats;
		self.num_T_x_inputs = 0;
		self.prior_family = multivariate_normal(D, T);
		self.prior = prior;
		self.data_num_resps = None;
		self.train_set = None;
		self.test_set = None;

	def get_efn_dims(self, param_net_input_type='eta', give_hint=False):
		"""Returns EFN component dimensionalities for the family.

		Args:
			param_net_input_type (str): specifies input to param network
				'eta':        give full eta to parameter network
				'prior':      part of eta that is prior-dependent
				'likelihood': part of eta that is likelihood-dependent
				'data':       the data itself
			give_hint: (bool): Feed in prior covariance cholesky if true.

		Returns:
			D_Z (int): dimensionality of density network
			num_suff_stats: dimensionality of eta
			num_param_net_inputs: dimensionality of param net input
			num_T_x_inputs: dimensionality of suff stat comp input
		"""

		if (give_hint):
			param_net_inputs_from_prior = self.num_prior_suff_stats + int(self.D*(self.D+1)/2);
		else:
			param_net_inputs_from_prior = self.num_prior_suff_stats;

		if (param_net_input_type == 'eta'):
			num_param_net_inputs = param_net_inputs_from_prior \
			                       + self.num_likelihood_suff_stats;
		elif (param_net_input_type == 'prior'):
			num_param_net_inputs = param_net_inputs_from_prior;
		elif (param_net_input_type == 'likelihood'):
			num_param_net_inputs = self.num_likelihood_suff_stats;
		elif (param_net_input_type == 'data'):
			num_param_net_inputs = self.D;

		return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_x_inputs;


	def map_to_support(self, layers, num_theta_params):
		"""Augment density network with bijective mapping to support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.
		"""
		support_layer = ShiftLayer(name='ShiftLayer', dim=self.D_Z);
		num_theta_params += count_layer_params(support_layer);
		layers.append(support_layer);
		return layers, num_theta_params;


	def compute_suff_stats(self, X, Z_by_layer, T_x_input):
		"""Compute sufficient statistics of density network samples.

		Args:
			X (tf.tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_x_input (tf.tensor): Param-dependent input.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""

		assert(self.T == 1);
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		T_x_prior = self.prior_family.compute_suff_stats(X, Z_by_layer, T_x_input);
		const = -tf.ones((K,M,1), tf.float64);
		z = X[:,:,:,0];
		sum_exp_z = tf.expand_dims(tf.reduce_sum(tf.exp(X[:,:,:,0]), 2), 2);
		T_x = tf.concat((T_x_prior, const, z, sum_exp_z), 2);
		return T_x;

	def compute_log_base_measure(self, X):
		"""Compute log base measure of density network samples.

		Args:
			X (tf.tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_x_input (tf.tensor): Param-dependent input.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""

		assert(self.T == 1);
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		log_h_x = tf.zeros((K,M), dtype=tf.float64);
		return log_h_x;


	def load_data(self,):
		datadir = 'data/responses/';
		num_monkeys = 3;
		num_neurons = [83, 59, 105]
		num_oris = 12;
		num_trials = 200;
		N = sum(num_neurons*num_oris);
		X = np.zeros((N,self.D_Z,num_trials));
		ind = 0;
		for i in range(num_monkeys):
			monkey = i+1;
			neurons = num_neurons[i];
			for j in range(neurons):
				neuron = j+1;
				for k in range(num_oris):
					ori = k+1;
					M = sio.loadmat(datadir + 'spike_counts_monkey%d_neuron%d_ori%d.mat' % (monkey, neuron, ori));
					X[ind,:,:] = M['x'][:,:self.D_Z].T;
					resp_info = {'monkey':monkey, 'neuron':neuron, 'ori':ori};
					assert(ind == self.resp_info_to_ind(resp_info));
					ind = ind + 1;
		self.data = X;
		self.data_num_resps = X.shape[0];
		return X;

	def select_train_test_sets(self, num_test):
		if (not (isinstance(num_test, int) and num_test >= 0)):
			print('Number of test set samples must be a non-negative integer.');
			exit();
		elif (num_test > self.data_num_resps):
			print('Asked for %d samples in test set, but only %d total responses.' % (num_test, self.data_num_resps));
			exit();

		if (num_test == 0):
			self.test_set = [];
			self.train_set = range(self.data_num_resps);
		else:
			self.test_set = np.sort(np.random.choice(self.data_num_resps, num_test, False)).tolist();
			inds = range(self.data_num_resps);
			test_set_inds = [i in self.test_set for i in inds];
			train_set_inds = [not i for i in test_set_inds];
			self.train_set = list(compress(inds,train_set_inds));

		return self.train_set, self.test_set;

	def resp_info_to_ind(self, resp_info):
		monkey = resp_info['monkey'];
		neuron = resp_info['neuron'];
		ori = resp_info['ori'];
		num_neurons = [83, 59, 105]
		num_oris = 12;
		ind = sum(num_neurons[:(monkey-1)])*num_oris + (neuron-1)*num_oris + (ori-1);
		return ind;

	def draw_etas(self, K, param_net_input_type='eta', give_hint=False, train=True, resp_info=None):
		datadir = 'data/responses/';
		_, _, num_param_net_inputs, _ = self.get_efn_dims(param_net_input_type, give_hint);
		eta = np.zeros((K, self.num_suff_stats));
		param_net_inputs = np.zeros((K, num_param_net_inputs));
		T_x_input = np.zeros((K, self.num_T_x_inputs));
		Ts = .02;
		mean_log_FR = -2.5892;
		var_log_FR = 0.4424;
		mu = mean_log_FR*np.ones((self.D_Z,));
		tau = .025;
		Sigma = var_log_FR*get_GP_Sigma(tau, self.D_Z, Ts)
		if (isinstance(self.prior, dict)):
			N = self.prior['N'];
		else:
			N = 200;
		params = [];

		if (K==1 and (resp_info is not None)):
			data_sets = [self.resp_info_to_ind(resp_info)]
		else:
			if (train):
				data_set_inds = np.random.choice(len(self.train_set), K, False);
				data_sets = [self.train_set[data_set_inds[i]] for i in range(K)];
			else:
				data_set_inds = np.random.choice(len(self.test_set), K, False);
				data_sets = [self.test_set[data_set_inds[i]] for i in range(K)];
		for k in range(K):
			x = self.data[data_sets[k]];
			params_k = {'mu':mu, 'Sigma':Sigma, 'x':x, 'N':N, 'data_ind':data_sets[k]};
			params.append(params_k);
			eta[k,:], param_net_inputs[k,:] = self.mu_to_eta(params_k, param_net_input_type, give_hint);
			T_x_input[k,:] = self.mu_to_T_x_input(params_k);
		return eta, param_net_inputs, T_x_input, params;

	def mu_to_eta(self, params, param_net_input_type='eta', give_hint=False):
		"""Maps mean parameters (mu) of distribution to canonical parameters (eta).

		Args:
			params (dict): Mean parameters
			param_net_input_type (str): specifies input to param network
				'eta':        give full eta to parameter network
			give_hint: (bool): Feed in prior covariance cholesky if true.
		Returns:
			eta (np.array): Canonical parameters.
			param_net_input: Corresponding input to parameter network.
		"""
		
		mu = params['mu'];
		Sigma = params['Sigma'];
		x = params['x'];
		N = params['N'];
		x = x[:,:N];

		alpha, alpha_param_net_input = self.prior_family.mu_to_eta(params, param_net_input_type, give_hint);
		mu = np.expand_dims(mu, 1);
		log_A_0 = 0.5*(np.dot(mu.T, np.dot(np.linalg.inv(Sigma), mu)) + np.log(np.linalg.det(Sigma)));
		sumx = np.sum(x, 1);

		eta_from_prior = np.concatenate((alpha, log_A_0[0]), 0);
		eta_from_likelihood = np.concatenate((sumx, -np.array([N])), 0);
		eta = np.concatenate((eta_from_prior, eta_from_likelihood), 0);

		param_net_input_from_prior = np.concatenate((alpha_param_net_input, log_A_0[0]), 0);
		param_net_input_from_likelihood = np.concatenate((sumx, -np.array([N])), 0);
		param_net_input_full = np.concatenate((param_net_input_from_prior, param_net_input_from_likelihood), 0);

		if (param_net_input_type == 'eta'):
			param_net_input = param_net_input_full;
		elif (param_net_input_type == 'prior'):
			param_net_input = param_net_input_from_prior;
		elif (param_net_input_type == 'likelihood'):
			param_net_input = param_net_input_from_likelihood;
		elif (param_net_input_type == 'data'):
			assert(x.shape[1] == 1 and N == 1);
			param_net_input = x.T;
		return eta, param_net_input;

class surrogate_S_D(family):
	"""Maximum entropy distribution with smoothness (S) and dim (D) constraints.

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		D_Z (int): dimensionality of the density network
		num_suff_stats (int): total number of suff stats
		num_T_x_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""

	def __init__(self, D, T=1):
		"""multivariate_normal family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""
		super().__init__(D, T);
		self.name = 'S_D';
		self.T = T;
		self.D_Z = D;
		self.num_suff_stats = int(D+D*(D+1)/2)*T + D*int((T-1)*T/2);
		self.set_T_x_names();

	def get_efn_dims(self, param_net_input_type='eta', give_hint=False):
		"""Returns EFN component dimensionalities for the family.

		Args:
			param_net_input_type (str): specifies input to param network
				'eta':        give full eta to parameter network
			give_hint: (bool): Feed in covariance cholesky if true.

		Returns:
			D_Z (int): dimensionality of density network
			num_suff_stats: dimensionality of eta
			num_param_net_inputs: dimensionality of param net input
			num_T_x_inputs: dimensionality of suff stat comp input
		"""

		num_param_net_inputs = None;
		return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_x_inputs;

	def set_T_x_names(self,):
		self.T_x_names = [];
		self.T_x_names_tf = [];
		self.T_x_group_names = [];

		for d in range(self.D):
			for t1 in range(self.T):
				for t2 in range(t1+1,self.T):
					self.T_x_names.append('$x_{%d,%d}$ $x_{%d,%d}$' % (d+1, t1+1, d+1, t2+1));
					self.T_x_names_tf.append('x_%d,%d x_%d,%d' % (d+1, t1+1, d+1, t2+1));
					self.T_x_group_names.append('$x_{%d,t}$ $x_{%d,t+%d}$' % (d+1, d+1, int(abs(t2-t1))));

		for i in range(self.D):
			for t in range(self.T):
				self.T_x_names.append('$x_{%d,%d}$' % (i+1, t+1));
				self.T_x_names_tf.append('x_%d,%d' % (i+1, t+1));
				self.T_x_group_names.append('$x_{%d,t}$' % (i+1));

		for i in range(self.D):
			for j in range(i,self.D):
				for t in range(self.T):
					self.T_x_names.append('$x_{%d,%d}$ $x_{%d,%d}$' % (i+1, t+1, j+1, t+1));
					self.T_x_names_tf.append('x_%d,%d x_%d,%d' % (i+1, t+1, j+1, t+1));
					self.T_x_group_names.append('$x_{%d,t}$ $x_{%d,t}$' % (i+1, j+1));
		return None;

	def compute_suff_stats(self, X, Z_by_layer, T_x_input):
		"""Compute sufficient statistics of density network samples.

		Args:
			X (tf.tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_x_input (tf.tensor): Param-dependent input.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		T = X_shape[3];
		_T = tf.cast(T, tf.int32);
		# compute the (S) suff stats
		#cov_con_mask_T = np.triu(np.ones((T,T), dtype=np.bool_), 1);
		ones_mat = tf.ones((T, T));
		mask_T = tf.matrix_band_part(ones_mat, 0, -1) - tf.matrix_band_part(ones_mat, 0, 0)
		cov_con_mask_T = tf.cast(mask_T, dtype=tf.bool);
		XXT_KMDTT = tf.matmul(tf.expand_dims(X, 4), tf.expand_dims(X, 3));
		T_x_S_KMDTcov = tf.transpose(tf.boolean_mask(tf.transpose(XXT_KMDTT, [3,4,0,1,2]), cov_con_mask_T), [1, 2, 3, 0])
		T_x_S = tf.reshape(T_x_S_KMDTcov, [K, M, self.D*tf.cast(T*(T-1)/2, tf.int32)]);

		# compute the (D) suff stats
		cov_con_mask_D = np.triu(np.ones((self.D,self.D), dtype=np.bool_), 0);
		T_x_mean = tf.reshape(X, [K,M,self.D*T]);

		X_KMTD = tf.transpose(X, [0, 1, 3, 2]); # samps x D
		XXT_KMTDD = tf.matmul(tf.expand_dims(X_KMTD, 4), tf.expand_dims(X_KMTD, 3));
		T_x_cov_KMDcovT = tf.transpose(tf.boolean_mask(tf.transpose(XXT_KMTDD, [3,4,0,1,2]), cov_con_mask_D), [1, 2, 0, 3]);
		T_x_cov = tf.reshape(T_x_cov_KMDcovT, [K,M,tf.cast(self.D*(self.D+1)/2, tf.int32)*T]);
		T_x_D = tf.concat((T_x_mean, T_x_cov), axis=2);

		print('T_x_S');
		print(T_x_S.shape);
		print('T_x_D');
		print(T_x_D.shape);
		# collect suff stats
		T_x = tf.concat((T_x_S, T_x_D), axis=2);
		print('T_x');
		print(T_x.shape);

		return T_x;

	def compute_log_base_measure(self, X):
		"""Compute log base measure of density network samples."""
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		log_h_x = tf.ones((K,M), dtype=tf.float64);
		return log_h_x;

	def compute_mu(self, params):
		# compute (S) part of mu
		kernel = params['kernel'];
		mu = params['mu'];
		Sigma = params['Sigma'];
		ts = params['ts'];
		_T = ts.shape[0];
		autocov_dim = int(_T*(_T-1)/2);
		mu_S = np.zeros((int(self.D*autocov_dim),));
		autocovs = np.zeros((self.D, _T));
		if (kernel == 'SE'): # squared exponential
			taus = params['taus'];

		elif (kernel == 'AR1'):
			alphas = params['alphas'];
			steps = np.arange(self.T);
			autocovs = np.zeros((self.D, self.T));
			for i in range(self.D):
				autocovs[i,:] = Sigma[i,i]*(alphas[i]**steps);
		else:
			raise NotImplementedError();

		ind = 0;
		ds = np.zeros((autocov_dim,));
		for t1 in range(_T):
			for t2 in range(t1+1, _T):
				ds[ind] = ts[t2] - ts[t1];
				ind += 1;

		for i in range(self.D):
			if (kernel == 'SE'):
				mu_S[(i*autocov_dim):((i+1)*autocov_dim)] = Sigma[i,i]*np.exp(-np.square(ds) / (2*np.square(taus[i])));
		
		# compute (D) part of mu		
		mu_mu = np.reshape(np.tile(mu, [1, _T]), [self.D*_T]);
		mu_Sigma = np.zeros((int(self.D*(self.D+1)/2)),);
		ind = 0;
		for i in range(self.D):
			for j in range(i,self.D):
				mu_Sigma[ind] = Sigma[i,j] + mu[i]*mu[j];
				ind += 1;
		mu_Sigma = np.reshape(np.tile(np.expand_dims(mu_Sigma, 1), [1, _T]), [int(self.D*(self.D+1)/2)*_T]);

		mu_D = np.concatenate((mu_mu, mu_Sigma), 0);

		mu = np.concatenate((mu_S, mu_D), 0);

		return mu;

class surrogate_S_D_C(family):
	"""Maximum entropy distribution with smoothness (S) and dim (D) constraints.

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		D_Z (int): dimensionality of the density network
		num_suff_stats (int): total number of suff stats
		num_T_x_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""

	def __init__(self, D, T, Tps, Tcs):
		"""multivariate_normal family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""
		super().__init__(D, T);
		self.name = 'S_D_C';
		self.D_Z = D;

		C = len(Tcs);
		mu_S_len = 0.0
		for i in range(C):
			mu_S_len += int(Tcs[i]*(Tcs[i]-1)/2);
			if (i > 0):
				mu_S_len = mu_S_len - 1;
		mu_S_len = self.D*mu_S_len;

		T_no_EP = T - 2*(C-1);
		mu_D_len = (D + (D*(D+1)/2))*T_no_EP;

		mu_C_len = D*int((C-1)*C/2)

		self.num_suff_stats = int(mu_S_len + mu_D_len + mu_C_len);
		self.Tps = Tps;
		self.Tcs = Tcs;
		self.set_T_x_names();

	def get_efn_dims(self, param_net_input_type='eta', give_hint=False):
		"""Returns EFN component dimensionalities for the family.

		Args:
			param_net_input_type (str): specifies input to param network
				'eta':        give full eta to parameter network
			give_hint: (bool): Feed in covariance cholesky if true.

		Returns:
			D_Z (int): dimensionality of density network
			num_suff_stats: dimensionality of eta
			num_param_net_inputs: dimensionality of param net input
			num_T_x_inputs: dimensionality of suff stat comp input
		"""

		num_param_net_inputs = None;
		return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_x_inputs;

	def set_T_x_names(self,):
		self.T_x_names = [];
		self.T_x_names_tf = [];
		self.T_x_group_names = [];
		C = len(self.Tcs);
		count = 0;
		for i in range(C):
			Tc = self.Tcs[i];
			for d in range(self.D):
				for t1 in range(Tc):
					for t2 in range(t1+1,Tc):
						if (i > 0 and t1==0 and t2==(Tc-1)):
							continue;
						self.T_x_names.append('$x_{%d,%d,%d}$ $x_{%d,%d,%d}$' % (i+1, d+1, t1+1, i+1, d+1, t2+1));
						self.T_x_names_tf.append('x_%d,%d,%d x_%d,%d,%d' % (i+1, d+1, t1+1, i+1, d+1, t2+1));
						self.T_x_group_names.append('$x_{%d,%d,t}$ $x_{%d,%d,t+%d}$' % (i+1, d+1, i+1, d+1, int(abs(t2-t1))));
						count += 1;
		print(count);
		for d in range(self.D):
			for i in range(C):
				Tc = self.Tcs[i];
				if (i==0):
					ts = range(Tc);
				else:
					ts = range(1, Tc-1);
				for t in ts:
					self.T_x_names.append('$x_{%d,%d,%d}$' % (i+1,d+1, t+1));
					self.T_x_names_tf.append('x_%d,%d,%d' % (i+1,d+1, t+1));
					self.T_x_group_names.append('$x_{%d,%d,t}$' % (i+1,d+1));
					count += 1;
		print(count);
		for d1 in range(self.D):
			for d2 in range(d1,self.D):
				for i in range(C):
					Tc = self.Tcs[i];
					if (i==0):
						ts = range(Tc);
					else:
						ts = range(1, Tc-1);
					for t in ts:
						self.T_x_names.append('$x_{%d,%d,%d}$ $x_{%d,%d,%d}$' % (i+1, d1+1, t+1, i+1, d2+1, t+1));
						self.T_x_names_tf.append('x_%d,%d,%d x_%d,%d,%d' % (i+1, d1+1, t+1, i+1, d2+1, t+1));
						self.T_x_group_names.append('$x_{%d,%d,t}$ $x_{%d,%d,t}$' % (i+1, d1+1, i+1, d2+1));
						count += 1;

		for d in range(self.D):
			for i in range(C):
				for j in range(i+1, C):
					self.T_x_names.append('$\sum_t x_{%d,%d,t}$ $x_{%d,%d,t}$' % (i+1, d+1, j+1, d+1));
					self.T_x_names_tf.append('Sum_t x_%d,%d,t x_%d,%d,t' % (i+1, d+1, j+1, d+1));
					self.T_x_group_names.append('$\sum_t x_{%d,%d,t}$ $x_{%d,%d,t}$' % (i+1, d+1, j+1, d+1));
					count += 1;


		print(count);
		return None;

	def compute_suff_stats(self, X, Z_by_layer, T_x_input):
		"""Compute sufficient statistics of density network samples.

		Args:
			X (tf.tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_x_input (tf.tensor): Param-dependent input.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];

		# compute the (S) suff stats
		C = len(self.Tcs);
		t_ind = 0;
		T_x_Ss = [];
		for i in range(C):
			Tc = self.Tcs[i];
			X_Tc = tf.slice(X, [0,0,0,t_ind], [K,M,self.D,Tc]);
			t_ind = t_ind + Tc;
			cov_con_mask_T = np.triu(np.ones((Tc,Tc), dtype=np.bool_), 1);
			XXT_KMDTT = tf.matmul(tf.expand_dims(X_Tc, 4), tf.expand_dims(X_Tc, 3));
			T_x_S_KMDTcov = tf.transpose(tf.boolean_mask(tf.transpose(XXT_KMDTT, [3,4,0,1,2]), cov_con_mask_T), [1, 2, 3, 0])
			if (i > 0):
				T_x_S_KMDTcov = tf.concat((T_x_S_KMDTcov[:,:,:,:(Tc-2)], T_x_S_KMDTcov[:,:,:,(Tc-1):]), 3);
				T_x_S_i = tf.reshape(T_x_S_KMDTcov, [K, M, self.D*int(Tc*(Tc-1)/2 - 1)]); # remove repeated endpoint correlation
			else:
				T_x_S_i = tf.reshape(T_x_S_KMDTcov, [K, M, self.D*int(Tc*(Tc-1)/2)]);
			T_x_Ss.append(T_x_S_i);
		T_x_S = tf.concat(T_x_Ss, 2);

		X_no_EP = self.remove_extra_endpoints_tf(X);
		T_no_EP = self.T - 2*(C-1);
		# compute the (D) suff stats
		cov_con_mask_D = np.triu(np.ones((self.D,self.D), dtype=np.bool_), 0);
		T_x_mean = tf.reshape(X_no_EP, [K,M,self.D*T_no_EP]);

		X_KMTD = tf.transpose(X_no_EP, [0, 1, 3, 2]); # samps x D
		XXT_KMTDD = tf.matmul(tf.expand_dims(X_KMTD, 4), tf.expand_dims(X_KMTD, 3));
		T_x_cov_KMDcovT = tf.transpose(tf.boolean_mask(tf.transpose(XXT_KMTDD, [3,4,0,1,2]), cov_con_mask_D), [1, 2, 0, 3]);
		T_x_cov = tf.reshape(T_x_cov_KMDcovT, [K,M,int(self.D*(self.D+1)/2)*T_no_EP]);
		T_x_D = tf.concat((T_x_mean, T_x_cov), axis=2);

		# compute the (C) suff stats
		Tc0 = self.Tcs[0];
		cov_con_mask_C = np.triu(np.ones((C,C), dtype=np.bool_), 1);
		T_x_cov_Cs = [];
		for d in range(self.D):
			X_d = X[:,:,d,:];
			X_cs = []
			t_ind = 0;
			for i in range(C):
				Tc = self.Tcs[i];
				X_d_i = tf.slice(X_d, [0, 0, t_ind], [K, M, Tc0]);
				X_cs.append(tf.expand_dims(X_d_i, 2));
				t_ind = t_ind + Tc;
			X_KMCT_d = tf.concat(X_cs, 2)
			XXT_KMCC_d = tf.div(tf.matmul(X_KMCT_d, tf.transpose(X_KMCT_d, [0,1,3,2])), Tc0);
			T_x_cov_KMCcov = tf.transpose(tf.boolean_mask(tf.transpose(XXT_KMCC_d, [2,3,0,1]), cov_con_mask_C), [1, 2, 0]);
			T_x_cov_Cs.append(T_x_cov_KMCcov);
		T_x_C = tf.concat(T_x_cov_Cs, 2);


		print('T_x_S');
		print(T_x_S.shape);
		print('T_x_D');
		print(T_x_D.shape);
		print('T_x_C');
		print(T_x_C.shape);
		# collect suff stats
		T_x = tf.concat((T_x_S, T_x_D, T_x_C), axis=2);
		print('T_x');
		print(T_x.shape);

		return T_x;

	def compute_log_base_measure(self, X):
		"""Compute log base measure of density network samples."""
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		log_h_x = tf.ones((K,M), dtype=tf.float64);
		return log_h_x;

	def compute_mu(self, params):
		# compute (S) part of mu
		kernel = params['kernel'];
		mu = params['mu'];
		Sigma = params['Sigma'];
		ts = params['ts'];
		C = len(self.Tcs);
		max_Tp = max(self.Tps);
		mu_S_len = 0;
		for i in range(C):
			mu_S_len += int(self.Tcs[i]*(self.Tcs[i]-1)/2);
			if (i > 0):
				mu_S_len = mu_S_len - 1;

		mu_S = np.zeros((self.D*mu_S_len,));
		if (kernel == 'SE'): # squared exponential
			taus = params['taus'];

		ind = 0;
		for i in range(C):
			Tc = self.Tcs[i];
			Tp = self.Tps[i];
			ts_i = np.concatenate(ts[:(i+1)]);
			ts_i = np.concatenate([np.array([0.0]), ts_i, np.array([Tp])]);
			for d in range(self.D):
				for t1 in range(Tc):
					for t2 in range(t1+1,Tc):
						if (i > 0 and t1==0 and t2==(Tc-1)):
							continue;
						mu_S[ind] = Sigma[i,i]*np.exp(-np.square(ts_i[t2]-ts_i[t1]) / (2*np.square(taus[i])));
						ind = ind + 1;
		# compute (D) part of mu
		T_no_EP = self.T - 2*(C-1);
		mu_mu = np.reshape(np.tile(mu, [1, T_no_EP]), [self.D*T_no_EP]);
		mu_Sigma = np.zeros((int(self.D*(self.D+1)/2)),);
		ind = 0;
		for i in range(self.D):
			for j in range(i,self.D):
				mu_Sigma[ind] = Sigma[i,j] + mu[i]*mu[j];
				ind += 1;
		mu_Sigma = np.reshape(np.tile(np.expand_dims(mu_Sigma, 1), [1, T_no_EP]), [int(self.D*(self.D+1)/2)*T_no_EP]);

		mu_D = np.concatenate((mu_mu, mu_Sigma), 0);

		mu_C = params['mu_C'];
		
		mu = np.concatenate((mu_S, mu_D, mu_C), 0);
		return mu;

	def remove_extra_endpoints_tf(self, X):
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		C = len(self.Tcs);
		Xs = [];
		t_ind = 0;
		for i in range(C):
			Tc = self.Tcs[i];
			if (i==0):
				X_i = tf.slice(X, [0, 0, 0, 0], [K, M, self.D, Tc]);
			else:
				X_i = tf.slice(X, [0, 0, 0, t_ind+1], [K, M, self.D, Tc-2]);
			t_ind = t_ind + Tc;
			Xs.append(X_i);
		X_no_EP = tf.concat(Xs, 3);
		return X_no_EP;

class GP_Dirichlet(family):
	"""Maximum entropy distribution with smoothness (S) and dim (D) constraints.

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		D_Z (int): dimensionality of the density network
		num_suff_stats (int): total number of suff stats
		num_T_x_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""

	def __init__(self, D, T=1):
		"""multivariate_normal family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""
		super().__init__(D, T);
		self.name = 'GP_Dirichlet';
		self.T = T;
		self.D_Z = D - 1;
		self.num_suff_stats = D*T + D*int((T-1)*T/2);
		self.set_T_x_names();

	def get_efn_dims(self, param_net_input_type='eta', give_hint=False):
		"""Returns EFN component dimensionalities for the family.

		Args:
			param_net_input_type (str): specifies input to param network
				'eta':        give full eta to parameter network
			give_hint: (bool): Feed in covariance cholesky if true.

		Returns:
			D_Z (int): dimensionality of density network
			num_suff_stats: dimensionality of eta
			num_param_net_inputs: dimensionality of param net input
			num_T_x_inputs: dimensionality of suff stat comp input
		"""

		num_param_net_inputs = None;
		return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_x_inputs;

	def set_T_x_names(self,):
		self.T_x_names = [];
		self.T_x_names_tf = [];
		self.T_x_group_names = [];

		for d in range(self.D):
			for t1 in range(self.T):
				for t2 in range(t1+1,self.T):
					self.T_x_names.append('$x_{%d,%d}$ $x_{%d,%d}$' % (d+1, t1+1, d+1, t2+1));
					self.T_x_names_tf.append('x_%d,%d x_%d,%d' % (d+1, t1+1, d+1, t2+1));
					self.T_x_group_names.append('$x_{%d,t}$ $x_{%d,t+%d}$' % (d+1, d+1, int(abs(t2-t1))));

		for i in range(self.D):
			for t in range(self.T):
				self.T_x_names.append('$\log x_{%d,%d}$' % (i+1, t+1));
				self.T_x_names_tf.append('\log x_%d,%d' % (i+1, t+1));
				self.T_x_group_names.append('$\log x_{%d,t}$' % (i+1));

		return None;

	def compute_suff_stats(self, X, Z_by_layer, T_x_input):
		"""Compute sufficient statistics of density network samples.

		Args:
			X (tf.tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_x_input (tf.tensor): Param-dependent input.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		T = X_shape[3];
		_T = tf.cast(T, tf.int32);
		# compute the (S) suff stats
		#cov_con_mask_T = np.triu(np.ones((T,T), dtype=np.bool_), 1);
		ones_mat = tf.ones((T, T));
		mask_T = tf.matrix_band_part(ones_mat, 0, -1) - tf.matrix_band_part(ones_mat, 0, 0)
		cov_con_mask_T = tf.cast(mask_T, dtype=tf.bool);
		XXT_KMDTT = tf.matmul(tf.expand_dims(X, 4), tf.expand_dims(X, 3));
		T_x_S_KMDTcov = tf.transpose(tf.boolean_mask(tf.transpose(XXT_KMDTT, [3,4,0,1,2]), cov_con_mask_T), [1, 2, 3, 0])
		T_x_S = tf.reshape(T_x_S_KMDTcov, [K, M, self.D*tf.cast(T*(T-1)/2, tf.int32)]);

		# compute the (D) suff stats
		log_X = tf.log(X);
		T_x_D = tf.reshape(log_X, [K, M, self.D*T]);

		print('T_x_S');
		print(T_x_S.shape);
		print('T_x_D');
		print(T_x_D.shape);
		# collect suff stats
		T_x = tf.concat((T_x_S, T_x_D), axis=2);
		print('T_x');
		print(T_x.shape);

		return T_x;

	def compute_log_base_measure(self, X):
		"""Compute log base measure of density network samples."""
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		log_h_x = tf.ones((K,M), dtype=tf.float64);
		return log_h_x;

	def compute_mu(self, params):
		alpha = params['alpha'];
		alpha_0 = np.sum(alpha);

		mean = alpha / alpha_0;
		var = np.multiply(alpha, alpha_0 - alpha) / (np.square(alpha_0)*(alpha_0+1));

		# compute (S) part of mu
		kernel = params['kernel'];
		ts = params['ts'];
		_T = ts.shape[0];
		autocov_dim = int(_T*(_T-1)/2);
		mu_S = np.zeros((int(self.D*autocov_dim),));
		autocovs = np.zeros((self.D, _T));
		if (kernel == 'SE'): # squared exponential
			taus = params['taus'];

		ind = 0;
		ds = np.zeros((autocov_dim,));
		for t1 in range(_T):
			for t2 in range(t1+1, _T):
				ds[ind] = ts[t2] - ts[t1];
				ind += 1;

		for i in range(self.D):
			if (kernel == 'SE'):
				mu_S[(i*autocov_dim):((i+1)*autocov_dim)] = var[i]*np.exp(-np.square(ds) / (2*np.square(taus[i]))) + np.square(mean[i]);
		
		# compute (D) part of mu		
		phi_0 = psi(alpha_0);
		mu_alpha = psi(alpha) - phi_0;
		thing = np.tile(np.expand_dims(mu_alpha, 1), [1, _T]);
		mu_D = np.reshape(np.tile(np.expand_dims(mu_alpha, 1), [1, _T]), [self.D*_T]);

		mu = np.concatenate((mu_S, mu_D), 0);

		return mu;

	def map_to_support(self, layers, num_theta_params):
		"""Augment density network with bijective mapping to support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.
		"""
		support_layer = SimplexBijectionLayer();
		num_theta_params += count_layer_params(support_layer);
		layers.append(support_layer);
		return layers, num_theta_params;

class NoTimeCon_GP_Dirichlet(family):
	"""Maximum entropy distribution with smoothness (S) and dim (D) constraints.

	Attributes:
		D (int): dimensionality of the exponential family
		T (int): number of time points
		D_Z (int): dimensionality of the density network
		num_suff_stats (int): total number of suff stats
		num_T_x_inputs (int): number of param-dependent inputs to suff stat comp.
		                      (only necessary for hierarchical dirichlet)
	"""

	def __init__(self, D, T=1):
		"""multivariate_normal family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""
		super().__init__(D, T);
		self.name = 'GP_Dirichlet';
		self.T = T;
		self.D_Z = D - 1;
		self.num_suff_stats = D*T;
		self.set_T_x_names();

	def get_efn_dims(self, param_net_input_type='eta', give_hint=False):
		"""Returns EFN component dimensionalities for the family.

		Args:
			param_net_input_type (str): specifies input to param network
				'eta':        give full eta to parameter network
			give_hint: (bool): Feed in covariance cholesky if true.

		Returns:
			D_Z (int): dimensionality of density network
			num_suff_stats: dimensionality of eta
			num_param_net_inputs: dimensionality of param net input
			num_T_x_inputs: dimensionality of suff stat comp input
		"""

		num_param_net_inputs = None;
		return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_x_inputs;

	def set_T_x_names(self,):
		self.T_x_names = [];
		self.T_x_names_tf = [];
		self.T_x_group_names = [];

		for i in range(self.D):
			for t in range(self.T):
				self.T_x_names.append('$\log x_{%d,%d}$' % (i+1, t+1));
				self.T_x_names_tf.append('\log x_%d,%d' % (i+1, t+1));
				self.T_x_group_names.append('$\log x_{%d,t}$' % (i+1));

		return None;

	def compute_suff_stats(self, X, Z_by_layer, T_x_input):
		"""Compute sufficient statistics of density network samples.

		Args:
			X (tf.tensor): Density network samples.
			Z_by_layer (list): List of layer activations in density network.
			T_x_input (tf.tensor): Param-dependent input.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		T = X_shape[3];
		_T = tf.cast(T, tf.int32);

		# compute the (D) suff stats
		log_X = tf.log(X);
		T_x_D = tf.reshape(log_X, [K, M, self.D*T]);

		return T_x_D;

	def compute_log_base_measure(self, X):
		"""Compute log base measure of density network samples."""
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		log_h_x = tf.ones((K,M), dtype=tf.float64);
		return log_h_x;

	def compute_mu(self, params):
		alpha = params['alpha'];
		alpha_0 = np.sum(alpha);

		ts = params['ts'];
		_T = ts.shape[0];

		# compute (D) part of mu		
		phi_0 = psi(alpha_0);
		mu_alpha = psi(alpha) - phi_0;
		thing = np.tile(np.expand_dims(mu_alpha, 1), [1, _T]);
		mu_D = np.reshape(np.tile(np.expand_dims(mu_alpha, 1), [1, _T]), [self.D*_T]);

		mu = mu_D

		return mu;

	def map_to_support(self, layers, num_theta_params):
		"""Augment density network with bijective mapping to support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.
		"""
		print('IN THE FUNC mapping using simplex bijection!');
		support_layer = SimplexBijectionLayer();
		num_theta_params += count_layer_params(support_layer);
		layers.append(support_layer);
		return layers, num_theta_params;

