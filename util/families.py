import tensorflow as tf
import numpy as np
from efn_util import count_layer_params, truncated_multivariate_normal_rvs, get_GP_Sigma, \
                     drawPoissonCounts
from flows import LinearFlowLayer, StructuredSpinnerLayer, PlanarFlowLayer, \
                  TanhLayer, SimplexBijectionLayer, CholProdLayer, SoftPlusLayer
import scipy.stats
from scipy.special import gammaln

class family:
	def __init__(self, D, T=1):
		self.D = D;
		self.T = T;
		self.num_T_x_inputs = 0;

	def get_efn_dims(self, param_net_input_type='eta', give_hint=False):
		raise NotImplementedError();

	def map_to_support(self, layers, num_theta_params):
		return layers, num_theta_params;

	def compute_suff_stats(self, X, Z_by_layer, T_x_input):
		raise NotImplementedError();

	def compute_log_base_measure(self, X, Z_by_layer, T_x_input):
		raise NotImplementedError();

	def draw_etas(self, K, param_net_input_type='eta', give_hint=False):
		raise NotImplementedError();

	def params_to_eta(self, params, param_net_input_type='eta', give_hint='False'):
		raise NotImplementedError();

	def params_to_T_x_input(self, params):
		T_x_input = np.array([]);
		return T_x_input;

	def batch_diagnostics(self, K, sess, feed_dict, X, log_p_zs, costs, R2s, eta_draw_params, checkEntropy=False):
		_X, _log_p_zs, _costs, _R2s = sess.run([X, log_p_zs, costs, R2s], feed_dict);
		KLs = [];
		for k in range(K):
			log_p_zs_k = _log_p_zs[k,:];
			X_k = _X[k, :, :, 0]; # TODO update this for time series
			params_k = eta_draw_params[k];
			KL_k = self.approxKL(log_p_zs_k, X_k, params_k);
			KLs.append(KL_k);
			if (checkEntropy):
				self.checkEntropy(log_p_zs_k, params_k);
		return _costs, _R2s, KLs;

	def approxKL(self, log_Q, X, params):
		return np.nan;

	def approxEntropy(self, log_Q):
		return np.mean(-log_Q);

	def trueEntropy(self, params):
		return np.nan;

	def checkEntropy(self, log_Q, params):
		approxH = self.approxEntropy(log_Q);
		trueH = self.trueEntropy(params)
		if (not np.isnan(trueH)):
			print('model entropy / true entropy');
			print('%.2E / %.2E' % (approxH, trueH));
		else:
			print('model entropy');
			print('%.2E' % approxH);
		return None;

class posterior_family(family):
	def __init__(self, D, T=1):
		super().__init__(D,T);
		self.D_Z = None;
		self.num_prior_suff_stats = None;
		self.num_likelihood_suff_stats = None;
		self.num_suff_stats = None;

	def get_efn_dims(self, param_net_input_type='eta', give_hint=False):
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
	def __init__(self, D, T=1):
		super().__init__(D, T);
		self.name = 'normal';
		self.D_Z = D;
		self.num_suff_stats = int(D+D*(D+1)/2);

	def get_efn_dims(self, param_net_input_type='eta', give_hint=False):
		if (not param_net_input_type == 'eta'):
			raise NotImplementedError();

		if (give_hint):
			num_param_net_inputs = int(self.D + self.D*(self.D+1)); # <- figure out what this number is
		else:
			num_param_net_inputs = int(self.D + self.D*(self.D+1)/2);
		return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_x_inputs;

	def compute_suff_stats(self, X, Z_by_layer, T_x_input):
		assert(self.T == 1);
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		cov_con_mask = np.triu(np.ones((self.D,self.D), dtype=np.bool_), 0);
		X_flat = tf.reshape(tf.transpose(X, [0, 1, 3, 2]), [K, M, self.D]); # samps x D
		T_x_mean = X_flat;
		XXT = tf.matmul(tf.expand_dims(X_flat, 3), tf.expand_dims(X_flat, 2));
		T_x_cov = tf.transpose(tf.boolean_mask(tf.transpose(XXT, [2,3,0,1]), cov_con_mask), [1, 2, 0]);
		T_x = tf.concat((T_x_mean, T_x_cov), axis=2);
		return T_x;

	def compute_log_base_measure(self, X):
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
			eta[k,:], param_net_inputs[k,:] = self.params_to_eta(params_k, param_net_input_type, give_hint);
			T_x_input[k,:] = self.params_to_T_x_input(params_k);
		return eta, param_net_inputs, T_x_input, params;

	def params_to_eta(self, params, param_net_input_type='eta', give_hint='False'):
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

	def approxKL(self, log_Q, X, params):
		batch_size = X.shape[0];
		mu = params['mu'];
		Sigma = params['Sigma'];
		dist = scipy.stats.multivariate_normal(mean=mu, cov=Sigma);
		log_P = dist.logpdf(X);
		KL = np.mean(log_Q - log_P);
		return KL;

	def trueEntropy(self, params):
		mu = params['mu'];
		Sigma = params['Sigma'];
		dist = scipy.stats.multivariate_normal(mean=mu, cov=Sigma);
		H_true = dist.entropy();
		return H_true;

class dirichlet(family):
	def __init__(self, D, T=1):
		super().__init__(D, T);
		self.name = 'dirichlet';
		self.D_Z = D-1;
		self.num_suff_stats = D;

	def get_efn_dims(self, param_net_input_type='eta', give_hint=False):
		if (give_hint or (not param_net_input_type == 'eta')):
			raise NotImplementedError();
		num_param_net_inputs = self.D;			
		return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_x_inputs;

	def map_to_support(self, layers, num_theta_params):
		support_layer = SimplexBijectionLayer();
		num_theta_params += count_layer_params(support_layer);
		layers.append(support_layer);
		return layers, num_theta_params;

	def compute_suff_stats(self, X, Z_by_layer, T_x_input):
		assert(self.T == 1);
		X_shape = tf.shape(X);
		K = X_shape[0];
		M = X_shape[1];
		X_flat = tf.reshape(tf.transpose(X, [0, 1, 3, 2]), [K, M, self.D]); # samps x D
		T_x_log = tf.log(X_flat);
		T_x = T_x_log;
		return T_x;

	def compute_log_base_measure(self, X):
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
			eta[k,:], param_net_inputs[k,:] = self.params_to_eta(params_k, param_net_input_type, give_hint);
			T_x_input[k,:] = self.params_to_T_x_input(params_k);
		return eta, param_net_inputs, T_x_input, params;

	def params_to_eta(self, params, param_net_input_type='eta', give_hint=False):
		if (give_hint or (not param_net_input_type == 'eta')):
			raise NotImplementedError();
		alpha = params['alpha'];
		eta = alpha;
		param_net_input = alpha;
		return eta, param_net_input;

	# the old construct_flow is fine
	def approxKL(self, log_Q, X, params):
		batch_size = X.shape[0];
		alpha = params['alpha'];
		dist = scipy.stats.dirichlet(np.float64(alpha));
		X = np.float64(X);
		X = X / np.expand_dims(np.sum(X, 1), 1);
		log_P = dist.logpdf(X.T);
		KL = np.mean(log_Q - log_P);
		return KL;

	def trueEntropy(self, params):
		alpha = params['alpha'];
		dist = scipy.stats.dirichlet(np.float64(alpha));
		H_true = dist.entropy();
		return H_true;

class inv_wishart(family):
	def __init__(self, D, T=1):
		super().__init__(D, T);
		self.name = 'dirichlet';
		self.sqrtD = int(np.sqrt(D));
		self.D_Z = int(self.sqrtD*(self.sqrtD+1)/2)
		self.num_suff_stats = self.D_Z + 1;

	def get_efn_dims(self, param_net_input_type='eta', give_hint=False):
		if (not param_net_input_type == 'eta'):
			raise NotImplementedError();

		if (give_hint):
			num_param_net_inputs = 2*self.D_Z + 1;
		else:
			num_param_net_inputs = self.num_suff_stats;
			
		return self.D_Z, self.num_suff_stats, num_param_net_inputs, self.num_T_x_inputs;

	def map_to_support(self, layers, num_theta_params):
		support_layer = CholProdLayer();
		num_theta_params += count_layer_params(support_layer);
		layers.append(support_layer);
		return layers, num_theta_params;

	def compute_suff_stats(self, X, Z_by_layer, T_x_input):
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
			eta[k,:], param_net_inputs[k,:] = self.params_to_eta(params_k, param_net_input_type, give_hint);
			T_x_input[k,:] = self.params_to_T_x_input(params_k);
		return eta, param_net_inputs, T_x_input, params;

	def params_to_eta(self, params, param_net_input_type='eta', give_hint=False):
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

	# the old construct_flow is fine
	def approxKL(self, log_Q, X, params):
		batch_size = X.shape[0];
		Psi = params['Psi'];
		m = params['m'];
		X = np.reshape(X, [batch_size, self.sqrtD, self.sqrtD]);
		log_P = scipy.stats.invwishart.logpdf(np.transpose(X, [1,2,0]), m, Psi);
		KL = np.mean(log_Q - log_P);
		return KL;

class hierarchical_dirichlet(posterior_family):
	def __init__(self, D, T=1):
		super().__init__(D, T);
		self.name = 'hierarchical_dirichlet';
		self.D_Z = D-1;
		self.num_prior_suff_stats = D + 1;
		self.num_likelihood_suff_stats = D + 1;
		self.num_suff_stats = self.num_prior_suff_stats + self.num_likelihood_suff_stats;
		self.num_T_x_inputs = 1;

	def map_to_support(self, layers, num_theta_params):
		support_layer = SimplexBijectionLayer();
		num_theta_params += count_layer_params(support_layer);
		layers.append(support_layer);
		return layers, num_theta_params;

	def compute_suff_stats(self, X, Z_by_layer, T_x_input):
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
			eta[k,:], param_net_inputs[k,:] = self.params_to_eta(params_k, param_net_input_type, False);
			T_x_input[k,:] = self.params_to_T_x_input(params_k);
		return eta, param_net_inputs, T_x_input, params;

	def params_to_eta(self, params, param_net_input_type='eta', give_hint=False):
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

	def params_to_T_x_input(self, params):
		beta = params['beta'];
		T_x_input = np.array([beta]);
		return T_x_input;

class dirichlet_multinomial(posterior_family):
	def __init__(self, D, T=1):
		super().__init__(D, T);
		self.name = 'dirichlet_multinomial';
		self.D_Z = D-1;
		self.num_prior_suff_stats = D + 1;
		self.num_likelihood_suff_stats = D + 1;
		self.num_suff_stats = self.num_prior_suff_stats + self.num_likelihood_suff_stats;
		self.num_T_x_inputs = 0;

	def map_to_support(self, layers, num_theta_params):
		support_layer = SimplexBijectionLayer();
		num_theta_params += count_layer_params(support_layer);
		layers.append(support_layer);
		return layers, num_theta_params;

	def compute_suff_stats(self, X, Z_by_layer, T_x_input):
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
			eta[k,:], param_net_inputs[k,:] = self.params_to_eta(params_k, param_net_input_type, False);
			T_x_input[k,:] = self.params_to_T_x_input(params_k);
		return eta, param_net_inputs, T_x_input, params;

	def params_to_eta(self, params, param_net_input_type='eta', give_hint=False):
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
	def __init__(self, D, T=1):
		super().__init__(D, T);
		self.name = 'truncated_normal_poisson';
		self.D_Z = D;
		self.num_prior_suff_stats = int(D+D*(D+1)/2) + 1;
		self.num_likelihood_suff_stats = D + 1;
		self.num_suff_stats = self.num_prior_suff_stats + self.num_likelihood_suff_stats;
		self.num_T_x_inputs = 0;
		self.prior_family = multivariate_normal(D, T);

	def map_to_support(self, layers, num_theta_params):
		support_layer = SoftPlusLayer();
		num_theta_params += count_layer_params(support_layer);
		layers.append(support_layer);
		return layers, num_theta_params;

	def compute_suff_stats(self, X, Z_by_layer, T_x_input):
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
			eta[k,:], param_net_inputs[k,:] = self.params_to_eta(params_k, param_net_input_type, False);
			T_x_input[k,:] = self.params_to_T_x_input(params_k);
		return eta, param_net_inputs, T_x_input, params;

	def params_to_eta(self, params, param_net_input_type='eta', give_hint=False):
		
		mu = params['mu'];
		Sigma = params['Sigma'];
		x = params['x'];
		z = params['z'];
		N = params['N'];
		assert(N == x.shape[1]);

		alpha, alpha_param_net_input = self.prior_family.params_to_eta(params, param_net_input_type, give_hint);
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
