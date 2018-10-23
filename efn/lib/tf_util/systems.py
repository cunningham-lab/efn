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
from lib.tf_util.tf_util import count_layer_params
from lib.tf_util.flows import SoftPlusLayer, IntervalFlowLayer
import scipy.stats
from scipy.special import gammaln, psi
import scipy.io as sio
from itertools import compress
from util import tf_integrals as tf_integrals


def system_from_str(system_str):
    if system_str in ["null", "null_on_interval"]:
        return null_on_interval
    elif system_str in ["one_con", "one_con_on_interval"]:
        return one_con_on_interval
    elif system_str in ["two_con", "two_con_on_interval"]:
        return two_con_on_interval
    elif system_str in ["linear_1D"]:
        return linear_1D
    elif system_str in ["linear_2D"]:
        return linear_2D
    elif system_str in ["damped_harmonic_oscillator", "dho"]:
        return damped_harmonic_oscillator
    elif system_str in ["rank1_rnn"]:
        return RNN_rank1
    elif system_str in ["rank1_rnn_std"]:
        return RNN_rank1_std


class system:
    """Base class for exponential families.
	
	Exponential families differ in their sufficient statistics, base measures,
	supports, ane therefore their natural parametrization.  Derivatives of this 
	class are useful tools for learning exponential family models in tensorflow.

	Attributes:
			T (int): number of time points
			dt (float): time resolution of simulation
			behavior_str (str): determines sufficient statistics that characterize system
	"""

    def __init__(self, behavior_str, T, dt):
        """family constructor

		Args:
			T (int): number of time points
			dt (float): time resolution of simulation
			behavior_str (str): determines sufficient statistics that characterize system
	
		"""
        self.behavior_str = behavior_str
        self.T = T
        self.dt = dt

    def map_to_parameter_support(self, layers, num_theta_params):
        """Augment density network with bijective mapping to parameter support."""
        return layers, num_theta_params

    def compute_suff_stats(self, phi):
        """Compute sufficient statistics of density network samples."""
        raise NotImplementedError()

    def analytic_suff_stats(self, phi):
        """Compute closed form sufficient statistics."""
        raise NotImplementedError()

    def simulation_suff_stats(self, phi):
        """Compute sufficient statistics that require simulation."""
        raise NotImplementedError()

    def compute_mu(self, params):
        """No comment yet."""
        raise NotImplementedError()

    def center_suff_stats_by_mu(self, T_x, mu):
        """Center sufficient statistics by the mean parameters mu."""
        return T_x - np.expand_dims(np.expand_dims(mu, 0), 1)


class null_on_interval(system):
    """Null system.  D parameters no constraints.  
	   Solution should be uniform on interval.

	Attributes:
		D (int): parametric dimensionality
		a (float): beginning of interval
		b (float): end of interval
	"""

    def __init__(self, D, a=0, b=1):
        self.name = "null_on_interval"
        self.D = D
        self.T = 1
        self.dt = 0.001
        self.num_suff_stats = 0
        self.a = a
        self.b = b

    def compute_suff_stats(self, phi):
        """Compute sufficient statistics of density network samples.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""
        phi_shape = tf.shape(phi)
        K = phi_shape[0]
        M = phi_shape[1]
        return tf.zeros((K, M, 0), dtype=tf.float64)

    def compute_mu(self, behavior):
        return np.array([], dtype=np.float64)

    def map_to_parameter_support(self, layers, num_theta_params):
        """Augment density network with bijective mapping to parameter support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.
		"""
        support_layer = IntervalFlowLayer("IntervalFlowLayer", self.a, self.b)
        num_theta_params += count_layer_params(support_layer)
        layers.append(support_layer)
        return layers, num_theta_params


class one_con_on_interval(null_on_interval):
    """System with one constraint.  D parameters dim1 == dim2.  
	   Solution should be uniform on the plane.

	Attributes:
		D (int): parametric dimensionality
		a (float): beginning of interval
		b (float): end of interval
	"""

    def __init__(self, D, a=0, b=1):
        super().__init__(D, a, b)
        if (type(D) is not int) or (D < 2):
            print("Error: need at least two dimensions for plane on interval")
            raise ValueError
        self.name = "one_con_on_interval"
        self.num_suff_stats = 2

    def compute_suff_stats(self, phi):
        """Compute sufficient statistics of density network samples.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""
        phi_shape = tf.shape(phi)
        K = phi_shape[0]
        M = phi_shape[1]
        diff01 = phi[:, :, 0] - phi[:, :, 1]
        T_x = tf.concat((diff01, tf.square(diff01)), axis=2)
        return T_x

    def compute_mu(self, behavior):
        return np.array([0.0, 0.001], dtype=np.float64)

    def map_to_parameter_support(self, layers, num_theta_params):
        """Augment density network with bijective mapping to parameter support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.
		"""
        support_layer = IntervalFlowLayer("IntervalFlowLayer", self.a, self.b)
        num_theta_params += count_layer_params(support_layer)
        layers.append(support_layer)
        return layers, num_theta_params


class two_con_on_interval(null_on_interval):
    """System with two constraints.  D parameters dim1 == dim2, dim2 == dim3  
	   Solution should be uniform on the plane.

	Attributes:
		D (int): parametric dimensionality
		a (float): beginning of interval
		b (float): end of interval
	"""

    def __init__(self, D, a=0, b=1):
        super().__init__(D, a, b)
        if (type(D) is not int) or (D < 3):
            print("Error: need at least three dimensions for plane on interval")
            raise ValueError
        self.name = "two_con_on_interval"
        self.num_suff_stats = 4

    def compute_suff_stats(self, phi):
        """Compute sufficient statistics of density network samples.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""
        phi_shape = tf.shape(phi)
        K = phi_shape[0]
        M = phi_shape[1]
        diff01 = phi[:, :, 0] - phi[:, :, 1]
        diff12 = phi[:, :, 1] - phi[:, :, 2]
        T_x = tf.concat((diff01, diff12, tf.square(diff01), tf.square(diff12)), axis=2)
        return T_x

    def compute_mu(self, behavior):
        return np.array([0.0, 0.0, 0.001, 0.001], dtype=np.float64)

    def map_to_parameter_support(self, layers, num_theta_params):
        """Augment density network with bijective mapping to parameter support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.
		"""
        support_layer = IntervalFlowLayer("IntervalFlowLayer", self.a, self.b)
        num_theta_params += count_layer_params(support_layer)
        layers.append(support_layer)
        return layers, num_theta_params


class linear_1D(system):
    """Linear one-dimensional systems.

	Attributes:
		D (int): parametric dimensionality
		T (int): number of time points
		dt (float): time resolution of simulation
		behavior_str (str): determines sufficient statistics that characterize system
	"""

    def __init__(self, behavior_str, T, dt, init_conds):
        super().__init__(behavior_str, T, dt)
        self.name = "linear_1D"
        self.D = 1
        self.init_conds = init_conds
        self.num_suff_stats = 2

    def simulate(self, phi):
        phi_shape = tf.shape(phi)
        K = phi_shape[0]
        M = phi_shape[1]
        X = []
        X_t = self.init_conds[0] * tf.ones((K, M, 1, 1), dtype=tf.float64)
        X.append(X_t)
        for i in range(1, self.T):
            X_dot = tf.multiply(phi, X_t)
            X_next = X_t + self.dt * X_dot
            X.append(X_next)
            X_t = X_next
        return tf.concat(X, axis=3)

    def compute_suff_stats(self, phi):
        """Compute sufficient statistics of density network samples.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""
        if self.behavior_str == "steady_state":
            T_x = self.simulation_suff_stats(phi)
        else:
            raise NotImplementedError
        return T_x

    def analytic_suff_stats(self, phi):
        """Compute closed form sufficient statistics.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Analytic sufficient statistics of samples.
		"""
        if self.behavior_str == "steady_state":
            ss = (self.init_conds[0] * tf.exp(self.dt * (self.T - 1) * phi))[:, :, :, 0]
            ss = tf.clip_by_value(ss, -1e3, 1e3)
            T_x = tf.concat((ss, tf.square(ss)), 2)
        return T_x

    def simulation_suff_stats(self, phi):
        """Compute sufficient statistics that require simulation.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Simulation-derived sufficient statistics of samples.
		"""
        if self.behavior_str == "steady_state":
            X = self.simulate(phi)
            ss = X[:, :, :, -1]
            T_x = tf.concat((ss, tf.square(ss)), 2)
        return T_x

    def compute_mu(self, behavior):
        mu = behavior["mu"]
        Sigma = behavior["Sigma"]
        mu_mu = mu
        mu_Sigma = np.zeros((int(self.D * (self.D + 1) / 2)))
        ind = 0
        for i in range(self.D):
            for j in range(i, self.D):
                mu_Sigma[ind] = Sigma[i, j] + mu[i] * mu[j]
                ind += 1

        mu = np.concatenate((mu_mu, mu_Sigma), 0)
        return mu


class damped_harmonic_oscillator(system):
    """Linear one-dimensional systems.

	Attributes:
		D (int): parametric dimensionality
		T (int): number of time points
		dt (float): time resolution of simulation
		behavior_str (str): determines sufficient statistics that characterize system
	"""

    def __init__(self, behavior_str, T, dt, init_conds, bounds):
        super().__init__(behavior_str, T, dt)
        self.name = "damped_harmonic_oscillator"
        self.D = 3
        self.init_conds = init_conds
        self.num_suff_stats = 2 * T
        self.bounds = bounds

    def simulate(self, phi):
        """Compute sufficient statistics that require simulation.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Simulation-derived sufficient statistics of samples.
		"""
        phi_shape = tf.shape(phi)
        K = phi_shape[0]
        M = phi_shape[1]

        print("phi", phi.shape)
        k = phi[:, :, 0, :]
        c = phi[:, :, 1, :]
        m = phi[:, :, 2, :]

        w_0 = tf.sqrt(tf.divide(k, m))
        zeta = tf.divide(c, 2.0 * tf.sqrt(tf.multiply(m, k)))

        def dydt(y, t):
            y1 = y[0]
            y2 = y[1]

            y1_dot = y2
            y2_dot = -2.0 * tf.multiply(tf.multiply(w_0, zeta), y2) - tf.multiply(
                tf.square(w_0), y1
            )

            ydot = tf.stack([y1_dot, y2_dot])
            return ydot

        y0 = tf.concat(
            (
                self.init_conds[0] * tf.ones((1, K, M, 1), dtype=tf.float64),
                self.init_conds[1] * tf.ones((1, K, M, 1), dtype=tf.float64),
            ),
            axis=0,
        )
        t = np.linspace(0, self.dt * (self.T - 1), self.T)

        out = tf.contrib.integrate.odeint_fixed(dydt, y0, t, method="rk4")

        return tf.transpose(out[:, :, :, :, 0], [2, 3, 1, 0])
        # make it K x M x D (sys) x T

    def compute_suff_stats(self, phi):
        """Compute sufficient statistics of density network samples.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""
        if self.behavior_str == "trajectory":
            T_x = self.simulation_suff_stats(phi)
        else:
            raise NotImplementedError
        return T_x

    def simulation_suff_stats(self, phi):
        """Compute sufficient statistics that require simulation.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Simulation-derived sufficient statistics of samples.
		"""
        if self.behavior_str == "trajectory":
            XY = self.simulate(phi)
            X = tf.clip_by_value(XY[:, :, 0, :], -1e3, 1e3)
            T_x = tf.concat((X, tf.square(X)), 2)
        return T_x

    def compute_mu(self, behavior):
        mu = behavior["mu"]
        Sigma = behavior["Sigma"]
        mu_mu = mu
        mu_Sigma = np.square(mu_mu) + Sigma
        print(mu_mu.shape, mu_Sigma.shape)
        mu = np.concatenate((mu_mu, mu_Sigma), 0)
        return mu

    def map_to_parameter_support(self, layers, num_theta_params):
        """Augment density network with bijective mapping to parameter support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.
		"""
        support_layer = IntervalFlowLayer(
            "IntervalFlowLayer", self.bounds[0], self.bounds[1]
        )
        num_theta_params += count_layer_params(support_layer)
        layers.append(support_layer)
        return layers, num_theta_params


class linear_2D(system):
    """Linear two-dimensional systems.

	Attributes:
		D (int): parametric dimensionality
		T (int): number of time points
		dt (float): time resolution of simulation
		behavior_str (str): determines sufficient statistics that characterize system
	"""

    def __init__(self, behavior_str):
        self.behavior_str = behavior_str
        self.name = "linear_2D"
        self.D = 4
        self.dt = 0.001
        self.T = 1
        self.num_suff_stats = 6

    def compute_suff_stats(self, phi):
        """Compute sufficient statistics of density network samples.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""
        if self.behavior_str == "oscillation":
            T_x = self.analytic_suff_stats(phi)
        else:
            raise NotImplementedError
        return T_x

    def analytic_suff_stats(self, phi):
        """Compute closed form sufficient statistics.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Analytic sufficient statistics of samples.
		"""
        phi_shape = tf.shape(phi)
        K = phi_shape[0]
        M = phi_shape[1]

        tau = 0.100
        # 100ms
        print("phi", phi.shape)
        """
		w1 = phi[:,:,0,:];
		w2 = phi[:,:,1,:];
		w3 = phi[:,:,2,:];
		w4 = phi[:,:,3,:];

		a1 = (w1-1)/tau;
		a2 = w2/tau;
		a3 = w3/tau;
		a4 = (w4-1)/tau;
		"""

        a1 = phi[:, :, 0, :]
        a2 = phi[:, :, 1, :]
        a3 = phi[:, :, 2, :]
        a4 = phi[:, :, 3, :]

        beta = tf.complex(tf.square(a1 + a4) - 4 * (a1 * a4 + a2 * a3), np.float64(0.0))
        beta_sqrt = tf.sqrt(beta)
        real_common = tf.complex(0.5 * (a1 + a4), np.float64(0.0))
        if self.behavior_str == "oscillation":
            lambda_1 = real_common + 0.5 * beta_sqrt
            lambda_2 = real_common - 0.5 * beta_sqrt
            lambda_1_real = tf.real(lambda_1)
            lambda_2_real = tf.real(lambda_2)
            lambda_1_imag = tf.imag(lambda_1)
            moments = [
                lambda_1_real,
                tf.square(lambda_1_real),
                lambda_2_real,
                tf.square(lambda_2_real),
                lambda_1_imag,
                tf.square(lambda_1_imag),
            ]
            T_x = tf.concat(moments, 2)
        return T_x

    def compute_mu(self, behavior):
        mu = behavior["mu"]
        Sigma = behavior["Sigma"]
        mu_mu = mu
        mu_Sigma = np.square(mu_mu) + Sigma
        print(mu_mu.shape, mu_Sigma.shape)
        mu = np.array(
            [mu_mu[0], mu_Sigma[0], mu_mu[1], mu_Sigma[1], mu_mu[2], mu_Sigma[2]]
        )
        return mu


class RNN_rank1(system):
    """Rank-1 RNN with bistable states for low input magnitudes
	   See Fig. 2F - Mastrogiuseppe et. al. 2018

	Attributes:
		D (int): parametric dimensionality
		T (int): number of time points
		dt (float): time resolution of simulation
		behavior_str (str): determines sufficient statistics that characterize system
	"""

    def __init__(self, T, Ics_0, Ics_1, behavior_str):
        self.behavior_str = behavior_str
        self.name = "rank1_rnn"
        self.D = 2
        self.dt = 0.001
        self.eps = 0.8
        self.g = 0.8
        self.T = T
        self.Ics_0 = Ics_0
        self.Ics_1 = Ics_1
        self.num_suff_stats = 8

    def compute_suff_stats(self, phi):
        """Compute sufficient statistics of density network samples.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""
        if self.behavior_str == "bistable":
            T_x = self.simulation_suff_stats(phi)
        else:
            raise NotImplementedError
        return T_x

    def simulation_suff_stats(self, phi):
        """Compute sufficient statistics that require simulation.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Simulation-derived sufficient statistics of samples.
		"""
        if self.behavior_str == "bistable":
            sol = self.simulate(phi)
            sol_shape = tf.shape(sol)
            K = sol_shape[1]
            M = sol_shape[2]
            D = sol_shape[3]
            X = tf.clip_by_value(sol[2, :, :, :], -1e3, 1e3)
            X = tf.expand_dims(tf.reshape(tf.transpose(X, [1, 0, 2]), [M, K * D]), 0)
            T_x = tf.concat((X, tf.square(X)), 2)
        return T_x

    def simulate(self, phi):
        Mm_tf = phi[:, :, 0, :]
        Mn_tf = phi[:, :, 1, :]

        Mm_tf = tf.tile(Mm_tf, [2, 1, 2])
        Mn_tf = tf.tile(Mn_tf, [2, 1, 2])

        # Mm = 3.5      # Mean of m
        # Mn = 1.       # Mean of n
        Mi = 0.0  # Mean of I

        Sim = 1.0  # Std of m
        Sin = 1.0  # Std of n
        Sip = 1.0

        Sini = np.concatenate(
            (0.5 * np.ones((1, 1, 2)), 1.0 * np.ones((1, 1, 2))), axis=0
        )

        def consistent_solve(y, g, eps, T):
            y_1 = y[:, :, :, 0]
            y_2 = y[:, :, :, 1]
            y_3 = y[:, :, :, 2]
            for i in range(T):
                Sii = np.sqrt((Sini / Sin) ** 2 + Sip ** 2)

                mu = Mm_tf * y_3 + Mi
                new1 = g * g * tf_integrals.PhiSq(mu, y_2) + Sim ** 2 * y_3 ** 2
                new1 = new1 + Sii ** 2
                new2 = Mn_tf * tf_integrals.Phi(mu, y_2) + Sini * tf_integrals.Prime(
                    mu, y_2
                )

                y_new_1 = Mm_tf * new2 + Mi
                y_new_2 = (1 - eps) * y_2 + eps * new1
                y_new_3 = (1 - eps) * y_3 + eps * new2

                y_1 = y_new_1
                y_2 = y_new_2
                y_3 = y_new_3

            y_out = tf.stack([y_1, y_2, y_3], axis=0)
            return y_out

        Ics = np.concatenate(
            (np.expand_dims(self.Ics_0, 2), np.expand_dims(self.Ics_1, 2)), axis=2
        )
        Ics = np.tile(Ics, [2, 1, 1, 1])
        sol = consistent_solve(Ics, self.g, self.eps, self.T)

        out = sol
        return sol

    def compute_mu(self, behavior):
        mu = behavior["mu"]
        Sigma = behavior["Sigma"]
        mu_mu = mu
        mu_Sigma = np.square(mu_mu) + Sigma
        mu = np.concatenate((mu_mu, mu_Sigma), axis=0)
        return mu

    def map_to_parameter_support(self, layers, num_theta_params):
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


class RNN_rank1_std(system):
    """Rank-1 RNN with bistable states for low input magnitudes
	   See Fig. 2F - Mastrogiuseppe et. al. 2018

	Attributes:
		D (int): parametric dimensionality
		T (int): number of time points
		dt (float): time resolution of simulation
		behavior_str (str): determines sufficient statistics that characterize system
	"""

    def __init__(self, T, Ics_0, Ics_1, behavior_str):
        self.behavior_str = behavior_str
        self.name = "rank1_rnn"
        self.D = 4
        self.dt = 0.001
        self.eps = 0.8
        self.g = 0.8
        self.T = T
        self.Ics_0 = Ics_0
        self.Ics_1 = Ics_1
        self.num_suff_stats = 8

    def compute_suff_stats(self, phi):
        """Compute sufficient statistics of density network samples.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""
        if self.behavior_str == "bistable":
            T_x = self.simulation_suff_stats(phi)
        else:
            raise NotImplementedError
        return T_x

    def simulation_suff_stats(self, phi):
        """Compute sufficient statistics that require simulation.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Simulation-derived sufficient statistics of samples.
		"""
        if self.behavior_str == "bistable":
            sol = self.simulate(phi)
            sol_shape = tf.shape(sol)
            K = sol_shape[1]
            M = sol_shape[2]
            D = sol_shape[3]
            X = tf.clip_by_value(sol[2, :, :, :], -1e3, 1e3)
            X = tf.expand_dims(tf.reshape(tf.transpose(X, [1, 0, 2]), [M, K * D]), 0)
            T_x = tf.concat((X, tf.square(X)), 2)
        return T_x

    def simulate(self, phi):
        Mm_tf = phi[:, :, 0, :]
        Mn_tf = phi[:, :, 1, :]
        Sim_tf = phi[:, :, 2, :]
        Sin_tf = phi[:, :, 3, :]

        Mm_tf = tf.tile(Mm_tf, [2, 1, 2])
        Mn_tf = tf.tile(Mn_tf, [2, 1, 2])
        Sim_tf = tf.tile(Sim_tf, [2, 1, 2])
        Sin_tf = tf.tile(Sin_tf, [2, 1, 2])

        # Mm = 3.5      # Mean of m
        # Mn = 1.       # Mean of n
        Mi = 0.0  # Mean of I

        # Sim = 1.      # Std of m
        # Sin = 1.      # Std of n
        Sip = 1.0

        Sini = np.concatenate(
            (0.5 * np.ones((1, 1, 2)), 1.0 * np.ones((1, 1, 2))), axis=0
        )

        def consistent_solve(y, g, eps, T):
            y_1 = y[:, :, :, 0]
            y_2 = y[:, :, :, 1]
            y_3 = y[:, :, :, 2]
            for i in range(T):
                Sii = tf.sqrt((Sini / Sin_tf) ** 2 + Sip ** 2)

                mu = Mm_tf * y_3 + Mi
                new1 = g * g * tf_integrals.PhiSq(mu, y_2) + Sim_tf ** 2 * y_3 ** 2
                new1 = new1 + Sii ** 2
                new2 = Mn_tf * tf_integrals.Phi(mu, y_2) + Sini * tf_integrals.Prime(
                    mu, y_2
                )

                y_new_1 = Mm_tf * new2 + Mi
                y_new_2 = (1 - eps) * y_2 + eps * new1
                y_new_3 = (1 - eps) * y_3 + eps * new2

                y_1 = y_new_1
                y_2 = y_new_2
                y_3 = y_new_3

            y_out = tf.stack([y_1, y_2, y_3], axis=0)
            return y_out

        Ics = np.concatenate(
            (np.expand_dims(self.Ics_0, 2), np.expand_dims(self.Ics_1, 2)), axis=2
        )
        Ics = np.tile(Ics, [2, 1, 1, 1])
        sol = consistent_solve(Ics, self.g, self.eps, self.T)

        out = sol
        return sol

    def compute_mu(self, behavior):
        mu = behavior["mu"]
        Sigma = behavior["Sigma"]
        mu_mu = mu
        mu_Sigma = np.square(mu_mu) + Sigma
        mu = np.concatenate((mu_mu, mu_Sigma), axis=0)
        return mu

    def map_to_parameter_support(self, layers, num_theta_params):
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
