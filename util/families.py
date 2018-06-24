import tensorflow as tf
import numpy as np
from efn_util import count_layer_params, truncated_multivariate_normal_rvs, get_GP_Sigma, \
                     drawPoissonCounts
from flows import LinearFlowLayer, StructuredSpinnerLayer, PlanarFlowLayer, \
                  TanhLayer, SimplexBijectionLayer, CholProdLayer, SoftPlusLayer
import scipy.stats
from scipy.special import gammaln, psi

def family_from_str(exp_fam_str):
	if (exp_fam_str in ['normal', 'multivariate_normal']):
		return multivariate_normal;
	elif (exp_fam_str in ['dirichlet']):
		return dirichlet;
	elif (exp_fam_str in ['inv_wishart']):
		return inv_wishart;
	elif (exp_fam_str in ['hierarchical_dirichlet', 'dir_dir']):
		return hierarchical_dirichlet;
	elif (exp_fam_str in ['dirichlet_multinomial']):
		return dirichlet_multinomial;
	elif (exp_fam_str in ['truncated_normal_poisson', 'prp_tn']):
		return truncated_normal_poisson;
	elif (exp_fam_str in ['S_D', 'D_S']):
		return surrogate_S_D;
	elif (exp_fam_str in ['S_D_nodyn', 'D_S_nodyn']):
		return surrogate_S_D_nodyn;
	elif (exp_fam_str in ['S_D_C', 'S_C_D', 'D_S_C', 'D_C_S', 'C_S_D', 'C_D_S']):
		return surrogate_S_D_C;

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

	def center_suff_stats_by_mu(self, T_x, params):
		"""Center sufficient statistics by the mean parameters mu."""
		raise NotImplementedError();

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
		return _elbos, _R2s, KLs;

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

	def center_suff_stats_by_mu(self, T_x, mu):
		"""Center sufficient statistics by the mean parameters mu."""
		return T_x - np.expand_dims(np.expand_dims(mu, 0), 1);


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

	def center_suff_stats_by_mu(self, T_x, mu):
		"""Center sufficient statistics by the mean parameters mu."""
		print(T_x.shape, mu.shape);
		return T_x - np.expand_dims(np.expand_dims(mu, 0), 1);

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

		alpha = params['alpha'];
		dist = scipy.stats.dirichlet(np.float64(alpha));
		X = np.float64(X);
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
		self.name = 'dirichlet';
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
		Nmax = 15;
		Nmean = 5;
		x_eps = 1e-16;
		params = [];
		for k in range(K):
			alpha_0_k = np.random.uniform(1.0, 10.0, (self.D,));
			beta_k = np.random.uniform(self.D, 2*self.D);
			N = int(min(np.random.poisson(Nmean), Nmax));
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
		Nmax = 50;
		Ts = .02;
		mu = 0.2*np.ones((self.D_Z,));
		tau = .025;
		Sigma = 0.26*get_GP_Sigma(tau, self.D_Z, Ts)
		params = [];
		for k in range(K):
			N = np.random.randint(1,Nmax+1);
			z = truncated_multivariate_normal_rvs(mu, Sigma);
			x = drawPoissonCounts(z, N);
			params_k = {'mu':mu, 'Sigma':Sigma, 'x':x, 'z':z, 'N':N};
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
		z = params['z'];
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

	def __init__(self, D, T=1, T_s=.001):
		"""multivariate_normal family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""
		super().__init__(D, T);
		self.name = 'S_D';
		self.D_Z = D;
		self.num_suff_stats = int(D+D*(D+1)/2)*T + D*int((T-1)*T/2);
		self.T_s = T_s;
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

		# compute the (S) suff stats
		cov_con_mask_T = np.triu(np.ones((self.T,self.T), dtype=np.bool_), 1);
		XXT_KMDTT = tf.matmul(tf.expand_dims(X, 4), tf.expand_dims(X, 3));
		T_x_S_KMDTcov = tf.transpose(tf.boolean_mask(tf.transpose(XXT_KMDTT, [3,4,0,1,2]), cov_con_mask_T), [1, 2, 3, 0])
		T_x_S = tf.reshape(T_x_S_KMDTcov, [K, M, self.D*int(self.T*(self.T-1)/2)]);

		# compute the (D) suff stats
		cov_con_mask_D = np.triu(np.ones((self.D,self.D), dtype=np.bool_), 0);
		T_x_mean = tf.reshape(X, [K,M,self.D*self.T]);

		X_KMTD = tf.transpose(X, [0, 1, 3, 2]); # samps x D
		XXT_KMTDD = tf.matmul(tf.expand_dims(X_KMTD, 4), tf.expand_dims(X_KMTD, 3));
		T_x_cov_KMDcovT = tf.transpose(tf.boolean_mask(tf.transpose(XXT_KMTDD, [3,4,0,1,2]), cov_con_mask_D), [1, 2, 0, 3]);
		T_x_cov = tf.reshape(T_x_cov_KMDcovT, [K,M,int(self.D*(self.D+1)/2)*self.T]);
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
		mu_S = np.zeros((int(self.D*self.T*(self.T-1)/2),));
		autocovs = np.zeros((self.D, self.T));
		if (kernel == 'SE'): # squared exponential
			taus = params['taus'];
			steps = np.arange(self.T)*self.T_s;
			for i in range(self.D):
				autocovs[i,:] = Sigma[i,i]*np.exp(-np.square(steps) / (2*np.square(taus[i])));

		elif (kernel == 'AR1'):
			alphas = params['alphas'];
			steps = np.arange(self.T);
			autocovs = np.zeros((self.D, self.T));
			for i in range(self.D):
				autocovs[i,:] = Sigma[i,i]*(alphas[i]**steps);
		else:
			raise NotImplementedError();

		ind = 0;
		for i in range(self.D):
			for t1 in range(self.T):
				for t2 in range(t1+1,self.T):
					mu_S[ind] = autocovs[i,t2-t1]
					ind = ind + 1;
		
		print('mu_S');
		print(mu_S.shape);

		# compute (D) part of mu		
		mu_mu = np.reshape(np.tile(mu, [1, self.T]), [self.D*self.T]);
		mu_Sigma = np.zeros((int(self.D*(self.D+1)/2)),);
		ind = 0;
		for i in range(self.D):
			for j in range(i,self.D):
				mu_Sigma[ind] = Sigma[i,j] + mu[i]*mu[j];
				ind += 1;
		mu_Sigma = np.reshape(np.tile(np.expand_dims(mu_Sigma, 1), [1, self.T]), [int(self.D*(self.D+1)/2)*self.T]);

		mu_D = np.concatenate((mu_mu, mu_Sigma), 0);

		print('mu_D');
		print(mu_D.shape);

		mu = np.concatenate((mu_S, mu_D), 0);
		print('mu');
		print(mu.shape);
		return mu;

	def center_suff_stats_by_mu(self, T_x, mu):
		"""Center sufficient statistics by the mean parameters mu."""
		return T_x - np.expand_dims(np.expand_dims(mu, 0), 1);


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

	def __init__(self, D, T, Tps, T_s=.001):
		"""multivariate_normal family constructor

		Args:
			D (int): dimensionality of the exponential family
			T (int): number of time points. Defaults to 1.
		"""
		super().__init__(D, T);
		self.name = 'S_D';
		self.D_Z = D;

		num_Tps = len(Tps);
		mu_S_len = 0.0
		for i in range(num_Tps):
			mu_S_len += int(Tps[i]*(Tps[i]-1)/2);
			if (i > 0):
				mu_S_len = mu_S_len - 1;
		mu_S_len = 2*mu_S_len;

		T_no_EP = T - 2*(num_Tps-1);
		mu_D_len = (D + (D*(D+1)/2))*T_no_EP;

		self.num_suff_stats = int(mu_S_len + mu_D_len);
		self.Tps = Tps;
		self.T_s = T_s;
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
		num_Tps = len(self.Tps);
		count = 0;
		for i in range(num_Tps):
			Tp = self.Tps[i];
			for d in range(self.D):
				for t1 in range(Tp):
					for t2 in range(t1+1,Tp):
						if (i > 0 and t1==0 and t2==(Tp-1)):
							continue;
						self.T_x_names.append('$x_{%d,%d,%d}$ $x_{%d,%d,%d}$' % (i+1, d+1, t1+1, i+1, d+1, t2+1));
						self.T_x_names_tf.append('x_%d,%d,%d x_%d,%d,%d' % (i+1, d+1, t1+1, i+1, d+1, t2+1));
						self.T_x_group_names.append('$x_{%d,%d,t}$ $x_{%d,%d,t+%d}$' % (i+1, d+1, i+1, d+1, int(abs(t2-t1))));
						count += 1;
		print(count);
		for d in range(self.D):
			for i in range(num_Tps):
				Tp = self.Tps[i];
				if (i==0):
					ts = range(Tp);
				else:
					ts = range(1, Tp-1);
				for t in ts:
					self.T_x_names.append('$x_{%d,%d,%d}$' % (i+1,d+1, t+1));
					self.T_x_names_tf.append('x_%d,%d,%d' % (i+1,d+1, t+1));
					self.T_x_group_names.append('$x_{%d,%d,t}$' % (i+1,d+1));
					count += 1;
		print(count);
		for d1 in range(self.D):
			for d2 in range(d1,self.D):
				for i in range(num_Tps):
					Tp = self.Tps[i];
					if (i==0):
						ts = range(Tp);
					else:
						ts = range(1, Tp-1);
					for t in ts:
						self.T_x_names.append('$x_{%d,%d,%d}$ $x_{%d,%d,%d}$' % (i+1, d1+1, t+1, i+1, d2+1, t+1));
						self.T_x_names_tf.append('x_%d,%d,%d x_%d,%d,%d' % (i+1, d1+1, t+1, i+1, d2+1, t+1));
						self.T_x_group_names.append('$x_{%d,%d,t}$ $x_{%d,%d,t}$' % (i+1, d1+1, i+1, d2+1));
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
		num_Tps = len(self.Tps);
		t_ind = 0;
		T_x_Ss = [];
		for i in range(num_Tps):
			Tp = self.Tps[i];
			X_Tp = tf.slice(X, [0,0,0,t_ind], [K,M,self.D,Tp]);
			t_ind = t_ind + Tp;
			cov_con_mask_T = np.triu(np.ones((Tp,Tp), dtype=np.bool_), 1);
			XXT_KMDTT = tf.matmul(tf.expand_dims(X_Tp, 4), tf.expand_dims(X_Tp, 3));
			T_x_S_KMDTcov = tf.transpose(tf.boolean_mask(tf.transpose(XXT_KMDTT, [3,4,0,1,2]), cov_con_mask_T), [1, 2, 3, 0])
			if (i > 0):
				T_x_S_KMDTcov = tf.concat((T_x_S_KMDTcov[:,:,:,:(Tp-2)], T_x_S_KMDTcov[:,:,:,(Tp-1):]), 3);
				T_x_S_i = tf.reshape(T_x_S_KMDTcov, [K, M, self.D*int(Tp*(Tp-1)/2 - 1)]); # remove repeated endpoint correlation
			else:
				T_x_S_i = tf.reshape(T_x_S_KMDTcov, [K, M, self.D*int(Tp*(Tp-1)/2)]);
			print(i, T_x_S_i.shape);
			T_x_Ss.append(T_x_S_i);
		T_x_S = tf.concat(T_x_Ss, 2);

		X_no_EP = self.remove_extra_endpoints_tf(X);
		T_no_EP = self.T - 2*(num_Tps-1);
		# compute the (D) suff stats
		cov_con_mask_D = np.triu(np.ones((self.D,self.D), dtype=np.bool_), 0);
		T_x_mean = tf.reshape(X_no_EP, [K,M,self.D*T_no_EP]);

		X_KMTD = tf.transpose(X_no_EP, [0, 1, 3, 2]); # samps x D
		XXT_KMTDD = tf.matmul(tf.expand_dims(X_KMTD, 4), tf.expand_dims(X_KMTD, 3));
		T_x_cov_KMDcovT = tf.transpose(tf.boolean_mask(tf.transpose(XXT_KMTDD, [3,4,0,1,2]), cov_con_mask_D), [1, 2, 0, 3]);
		T_x_cov = tf.reshape(T_x_cov_KMDcovT, [K,M,int(self.D*(self.D+1)/2)*T_no_EP]);
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
		num_Tps = len(self.Tps);
		max_Tp = max(self.Tps);
		mu_S_len = 0;
		for i in range(num_Tps):
			mu_S_len += int(self.Tps[i]*(self.Tps[i]-1)/2);
			if (i > 0):
				mu_S_len = mu_S_len - 1;

		mu_S = np.zeros((self.D*mu_S_len,));
		autocovs = np.zeros((self.D, max_Tp));
		if (kernel == 'SE'): # squared exponential
			taus = params['taus'];
			
			steps = np.arange(max_Tp)*self.T_s;
			for i in range(self.D):
				autocovs[i,:] = Sigma[i,i]*np.exp(-np.square(steps) / (2*np.square(taus[i])));

		elif (kernel == 'AR1'):
			alphas = params['alphas'];
			steps = np.arange(self.T);
			autocovs = np.zeros((self.D, self.T));
			for i in range(self.D):
				autocovs[i,:] = Sigma[i,i]*(alphas[i]**steps);
		else:
			raise NotImplementedError();

		ind = 0;
		for d in range(self.D):
			for i in range(num_Tps):
				Tp = self.Tps[i];
				for t1 in range(Tp):
					for t2 in range(t1+1,Tp):
						if (i > 0 and t1==0 and t2==(Tp-1)):
							continue;
						mu_S[ind] = autocovs[d,t2-t1]
						ind = ind + 1;
		
		print('mu_S');
		print(mu_S.shape);

		# compute (D) part of mu
		T_no_EP = self.T - 2*(num_Tps-1);
		mu_mu = np.reshape(np.tile(mu, [1, T_no_EP]), [self.D*T_no_EP]);
		mu_Sigma = np.zeros((int(self.D*(self.D+1)/2)),);
		ind = 0;
		for i in range(self.D):
			for j in range(i,self.D):
				mu_Sigma[ind] = Sigma[i,j] + mu[i]*mu[j];
				ind += 1;
		mu_Sigma = np.reshape(np.tile(np.expand_dims(mu_Sigma, 1), [1, T_no_EP]), [int(self.D*(self.D+1)/2)*T_no_EP]);

		mu_D = np.concatenate((mu_mu, mu_Sigma), 0);

		print('mu_D');
		print(mu_D.shape);

		mu = np.concatenate((mu_S, mu_D), 0);
		print('mu');
		print(mu.shape);
		return mu;

	def center_suff_stats_by_mu(self, T_x, mu):
		"""Center sufficient statistics by the mean parameters mu."""
		return T_x - np.expand_dims(np.expand_dims(mu, 0), 1);

	def remove_extra_endpoints_tf(self, X):
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		num_Tps = len(self.Tps);
		Xs = [];
		t_ind = 0;
		for i in range(num_Tps):
			Tp = self.Tps[i];
			if (i==0):
				X_i = tf.slice(X, [0, 0, 0, 0], [K, M, self.D, Tp]);
			else:
				X_i = tf.slice(X, [0, 0, 0, t_ind+1], [K, M, self.D, Tp-2]);
			t_ind = t_ind + Tp;
			Xs.append(X_i);
		X_no_EP = tf.concat(Xs, 3);
		return X_no_EP;



