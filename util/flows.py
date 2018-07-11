import tensorflow as tf
import numpy as np
import scipy.linalg

# Time invariant flows

def efn_tensor_shape(z):
    # could use eager execution to iterate over shape instead.  Avoiding for now.
    K = tf.shape(z)[0];
    M = tf.shape(z)[1];
    D = tf.shape(z)[2];
    T = tf.shape(z)[3];
    return K, M, D, T;

class Layer:
    def __init__(self, name=''):
        self.name = name;
        self.param_names = [];
        self.param_network = False;

    def get_layer_info(self,):
        return self.name, [], [], [], None;

    def get_params(self,):
        return [];

    def connect_parameter_network(self, theta_layer):
        return None;
    def forward_and_jacobian(self, y):
        raise NotImplementedError(str(type(self)));
        
class PlanarFlowLayer(Layer):

    def __init__(self, name='PlanarFlow', dim=1):
        # TODO this entire class needs to be compatible with the parameter network
        self.name = name;
        self.dim = dim;
        self.param_names = ['u', 'w', 'b'];
        self.param_network = False;
        self.u = None;
        self.uhat = None;
        self.w = None;
        self.b = None;
        self.lock = False;

    def get_layer_info(self,):
        u_dim = (self.dim,1);
        w_dim = (self.dim,1);
        b_dim = (1,1);
        dims = [u_dim, w_dim, b_dim];
        initializers = [tf.constant(np.zeros(u_dim)), tf.glorot_uniform_initializer(), \
                        tf.constant(np.zeros(b_dim))];
        return self.name, self.param_names, dims, initializers, self.lock;

    def get_params(self,):
        if (not self.param_network):
            return tf.expand_dims(self.uhat, 0), tf.expand_dims(self.w, 0), tf.expand_dims(self.b, 0);
        else:
            return self.uhat, self.w, self.b;
        
    def connect_parameter_network(self, theta_layer):
        self.u, self.w, self.b = theta_layer;
        self.param_network = (len(self.u.shape) == 3);
        if (self.param_network):
            wdotu = tf.matmul(tf.transpose(self.w, [0, 2, 1]), self.u);
            mwdotu = -1 + tf.log(1 + tf.exp(wdotu));
            self.uhat = self.u + tf.divide(tf.multiply((mwdotu - wdotu),self.w), tf.expand_dims(tf.reduce_sum(tf.square(self.w),1), 1));
        else:
            wdotu = tf.matmul(tf.transpose(self.w), self.u);
            mwdotu = -1 + tf.log(1 + tf.exp(wdotu));
            self.uhat = self.u + tf.divide((mwdotu - wdotu)*self.w, tf.reduce_sum(tf.square(self.w)));

        return None;
            
    def forward_and_jacobian(self, z, sum_log_det_jacobians, reuse=False):
        u, w, b = self.get_params();
        K, M, D, T = efn_tensor_shape(z);

        z_KD_MTvec = tf.reshape(tf.transpose(z, [0,2,1,3]), [K,D,M*T]);
        # helper function phi_k(z_{k-1})
        # derivative of tanh(x) is (1-tanh^2(x))
        phi = tf.matmul(w, (1.0 - tf.tanh(tf.matmul(tf.transpose(w, [0,2,1]), z_KD_MTvec) + b)**2));
        # compute the running sum of log-determinant of jacobians
        log_det_jacobian = tf.log(tf.abs(1.0 + tf.matmul(tf.transpose(u, [0,2,1]), phi)));
        log_det_jacobian = tf.reshape(log_det_jacobian, [K,M,T]); 
        log_det_jacobian = tf.reduce_sum(log_det_jacobian, 2);

        # compute z for this layer
        nonlin_term = tf.tanh(tf.matmul(tf.transpose(w, [0,2,1]), z_KD_MTvec) + b);
        z = z + tf.transpose(tf.reshape(tf.matmul(u, nonlin_term), [K,D,M,T]), [0,2,1,3]);

        sum_log_det_jacobians += log_det_jacobian;

        return z, sum_log_det_jacobians;

class SimplexBijectionLayer(Layer):
    def __init__(self, name='SimplexBijectionLayer'):
        self.name = name;
        self.param_names = [];
        self.param_network = False;

    def forward_and_jacobian(self, z, sum_log_det_jacobians):
        D = tf.shape(z)[2];
        ex = tf.exp(z);
        den = tf.reduce_sum(ex, 2) + 1.0;
        log_dets = tf.log(1.0 - (tf.reduce_sum(ex, 2) / den)) - tf.cast(D, tf.float64)*tf.log(den) + tf.reduce_sum(z, 2);
        z = tf.concat((ex / tf.expand_dims(den, 2), 1.0 / tf.expand_dims(den, 2)), axis=2);
        sum_log_det_jacobians += tf.reduce_sum(log_dets, 2);
        return z, sum_log_det_jacobians;


class CholProdLayer(Layer):
    def __init__(self, name='SimplexBijectionLayer'):
        # TODO this entire class needs to be compatible with the parameter network
        self.name = name;
        self.param_names = [];
        self.param_network = False;

    def forward_and_jacobian(self, z, sum_log_det_jacobians):
        K, M, D_Z, T = efn_tensor_shape(z);
        z_KMD_Z = z[:,:,:,0]; # generalize this for more time points
        L = tf.contrib.distributions.fill_triangular(z_KMD_Z);
        sqrtD = tf.shape(L)[2];
        sqrtD_flt = tf.cast(sqrtD, tf.float64);
        D = tf.square(sqrtD);
        L_pos_diag = tf.contrib.distributions.matrix_diag_transform(L, tf.exp);
        LLT = tf.matmul(L_pos_diag, tf.transpose(L_pos_diag, [0,1,3,2]));
        #give it a lil boost
        #diag_boost = 10*p_eps*tf.eye(sqrtD, batch_shape=[K,M], dtype=tf.float64);
        #LLT = LLT + diag_boost;
        LLT_vec = tf.reshape(LLT, [K,M,D]);
        z = tf.expand_dims(LLT_vec, 3); # update this for T > 1
        
        L_diag_els = tf.matrix_diag_part(L);
        L_pos_diag_els = tf.matrix_diag_part(L_pos_diag);
        var = tf.cast(tf.range(1,sqrtD+1), tf.float64);
        pos_diag_support_log_det = tf.reduce_sum(L_diag_els, 2);
        #
        diag_facs = tf.expand_dims(tf.expand_dims(sqrtD_flt - var + 1.0, 0), 0);
        chol_prod_log_det = sqrtD_flt * np.log(2.0) + tf.reduce_sum(tf.multiply(diag_facs, tf.log(L_pos_diag_els)), 2);
        sum_log_det_jacobians += (pos_diag_support_log_det + chol_prod_log_det);

        return z, sum_log_det_jacobians;

class TanhLayer(Layer):
    def __init__(self, name='TanhLayer'):
        self.name = name;
        self.param_names = [];
        self.param_network = False;

    def forward_and_jacobian(self, z, sum_log_det_jacobians):
        z_out = tf.tanh(z);
        log_det_jacobian = tf.reduce_sum(tf.log(1.0 - (z_out**2)), [2,3]);
        sum_log_det_jacobians += log_det_jacobian;
        return z_out, sum_log_det_jacobians;

class ExpLayer(Layer):
    def __init__(self, name='ExpLayer'):
        self.name = name;
        self.param_names = [];
        self.param_network = False;

    def forward_and_jacobian(self, z, sum_log_det_jacobians):
        z_out = tf.exp(z);
        log_det_jacobian = tf.reduce_sum(z, [2,3]);
        sum_log_det_jacobians += log_det_jacobian;
        return z_out, sum_log_det_jacobians;

class SoftPlusLayer(Layer):
    def __init__(self, name='SoftPlusLayer'):
        self.name = name;
        self.param_names = [];
        self.param_network = False;

    def forward_and_jacobian(self, z, sum_log_det_jacobians):
        z_out = tf.log(1 + tf.exp(z));
        jacobian = tf.divide(1.0, 1.0 + tf.exp(-z));
        log_det_jacobian = tf.log(tf.reduce_prod(jacobian, 2,3));
        sum_log_det_jacobians += log_det_jacobian;
        return z_out, sum_log_det_jacobians;




class StructuredSpinnerLayer(Layer):
    def __init__(self, name, dim):
        self.name = name;
        self.dim = dim;
        self.param_names = ['d1', 'd2', 'd3', 'b'];
        self.d1 = None;
        self.d2 = None;
        self.d3 = None;
        self.b = None;
        self.lock = False;
        
    def get_layer_info(self,):
        d1_dim = (self.dim, 1);
        d2_dim = (self.dim, 1);
        d3_dim = (self.dim, 1);
        b_dim = (self.dim,1);
        dims = [d1_dim, d2_dim, d3_dim, b_dim];
        initializers = [tf.glorot_uniform_initializer(), tf.glorot_uniform_initializer(), \
                        tf.glorot_uniform_initializer(), tf.glorot_uniform_initializer()];
        return self.name, self.param_names, dims, initializers, self.lock;

    def get_params(self,):
        return self.d1, self.d2, self.d3, self.b;

    def connect_parameter_network(self, theta_layer):
        self.d1, self.d2, self.d3, self.b = theta_layer;
        # params will have batch dimension if result of parameter network
        self.param_network = (len(self.d1.shape) == 3);
        return None;
            
    def forward_and_jacobian(self, z, sum_log_det_jacobians):
        K, M, D, T = efn_tensor_shape(z);
        D_np = z.get_shape().as_list()[2];
        Hmat = scipy.linalg.hadamard(D_np, dtype=np.float64) / np.sqrt(D_np);
        H = tf.constant(Hmat , tf.float64);
        d1, d2 ,d3, b = self.get_params();
        D1 = tf.diag(d1[:,0]);
        D2 = tf.diag(d2[:,0]);
        D3 = tf.diag(d3[:,0]);
        A = tf.matmul(H, tf.matmul(D3, tf.matmul(H, tf.matmul(D2, tf.matmul(H, D1)))));

        if (not self.param_network):
            b = tf.expand_dims(tf.expand_dims(b, 0), 0);
            z = tf.tensordot(A, z, [[1], [2]]);
            z = tf.transpose(z, [1, 2, 0, 3]) + b;
            log_det_A = tf.log(tf.abs(tf.reduce_prod(d1))) + tf.log(tf.abs(tf.reduce_prod(d2))) + tf.log(tf.abs(tf.reduce_prod(d3)));
            log_det_jacobian = tf.multiply(log_det_A, tf.ones((K,M), dtype=tf.float64)) + 0.0*tf.reduce_sum(b);
        else:
            z_KD_MTvec = tf.reshape(tf.transpose(z, [0,2,1,3]), [K,D,M*T]);
            Az_KD_MTvec = tf.matmul(A, z_KD_MTvec);
            Az = tf.transpose(tf.reshape(Az_KD_MTvec, [K,D,M,T]), [0,2,1,3]);
            z = Az + tf.expand_dims(b, 1);
            log_det_jacobian = tf.log(tf.abs(tf.matrix_determinant(A)));
            log_det_jacobian = tf.tile(tf.expand_dims(log_det_jacobian, 1), [1, M]) + 0.0*tf.reduce_sum(b);
        sum_log_det_jacobians += tf.cast(T, tf.float64)*log_det_jacobian;
        return z, sum_log_det_jacobians;


class LinearFlowLayer(Layer):
    def __init__(self, name, dim):
        self.name = name;
        self.dim = dim;
        self.param_names = ['A', 'b'];
        self.lock = False;
        
    def get_layer_info(self,):
        A_dim = (self.dim, self.dim);
        b_dim = (self.dim,1);
        dims = [A_dim, b_dim];
        initializers = [tf.glorot_uniform_initializer(), tf.glorot_uniform_initializer()];
        return self.name, self.param_names, dims, initializers, self.lock;

    def connect_parameter_network(self, theta_layer):
        self.A = theta_layer[0];
        self.b = theta_layer[1];
        # params will have batch dimension if result of parameter network
        self.param_network = (len(self.A.shape) == 3);
        return None;
            
    def forward_and_jacobian(self, z, sum_log_det_jacobians):
        K, M, D, T = efn_tensor_shape(z);
        if (not self.param_network):
            A = self.A;
            b = tf.expand_dims(tf.expand_dims(self.b, 0), 0);
            z = tf.tensordot(A, z, [[1], [2]]);
            z = tf.transpose(z, [1, 2, 0, 3]) + b;
            log_det_jacobian = tf.multiply(tf.log(tf.abs(tf.matrix_determinant(A))), tf.ones((K,M,), dtype=tf.float64)) + 0.0*tf.reduce_sum(b);
        else:
            z_KD_MTvec = tf.reshape(tf.transpose(z, [0,2,1,3]), [K,D,M*T]);
            Az_KD_MTvec = tf.matmul(self.A, z_KD_MTvec);
            Az = tf.transpose(tf.reshape(Az_KD_MTvec, [K,D,M,T]), [0,2,1,3]);
            z = Az + tf.expand_dims(self.b, 1);
            log_det_jacobian = tf.log(tf.abs(tf.matrix_determinant(self.A)))
            log_det_jacobian = tf.tile(tf.expand_dims(log_det_jacobian, 1), [1, M]);
        sum_log_det_jacobians += tf.cast(T, tf.float64)*log_det_jacobian;
        return z, sum_log_det_jacobians;


class FullyConnectedFlowLayer(Layer):
    def __init__(self, name, dim, A_init, lock):
        self.name = name;
        self.dim = dim;
        self.param_names = ['A'];
        self.A_init = A_init;
        self.lock = lock;
        
    def get_layer_info(self,):
        A_dim = (self.dim, self.dim);
        dims = [A_dim];
        initializers = [tf.constant(self.A_init)];
        return self.name, self.param_names, dims, initializers, self.lock;

    def connect_parameter_network(self, theta_layer):
        self.A = theta_layer[0];
        # params will have batch dimension if result of parameter network
        self.param_network = (len(self.A.shape) == 3);
        return None;
            
    def forward_and_jacobian(self, z, sum_log_det_jacobians):
        K, M, D, T = efn_tensor_shape(z);
        if (not self.param_network):
            A = self.A;
            z = tf.tensordot(A, z, [[1], [2]]);
            z = tf.transpose(z, [1, 2, 0, 3]);
            log_det_jacobian = tf.multiply(tf.log(tf.abs(tf.matrix_determinant(A))), tf.ones((K,M,), dtype=tf.float64));
        else:
            z_KD_MTvec = tf.reshape(tf.transpose(z, [0,2,1,3]), [K,D,M*T]);
            Az_KD_MTvec = tf.matmul(self.A, z_KD_MTvec);
            Az = tf.transpose(tf.reshape(Az_KD_MTvec, [K,D,M,T]), [0,2,1,3]);
            z = Az;
            log_det_jacobian = tf.log(tf.abs(tf.matrix_determinant(self.A)))
            log_det_jacobian = tf.tile(tf.expand_dims(log_det_jacobian, 1), [1, M]);
        print(sum_log_det_jacobians.shape);
        print(log_det_jacobian.shape);
        sum_log_det_jacobians += log_det_jacobian;
        return z, sum_log_det_jacobians;


class ShiftLayer(Layer):
    def __init__(self, name, dim):
        self.name = name;
        self.dim = dim;
        self.param_names = ['b'];
        self.lock = False;
        
    def get_layer_info(self,):
        b_dim = (self.dim,1);
        dims = [b_dim];
        initializers = [tf.glorot_uniform_initializer()]
        return self.name, self.param_names, dims, initializers, self.lock;

    def connect_parameter_network(self, theta_layer):
        self.b = theta_layer[0];
        # params will have batch dimension if result of parameter network
        self.param_network = (len(self.b.shape) == 3);
        return None;
            
    def forward_and_jacobian(self, z, sum_log_det_jacobians):
        K, M, D, T = efn_tensor_shape(z);
        if (not self.param_network):
            b = tf.expand_dims(tf.expand_dims(self.b, 0), 0);
            z = z + b;
            log_det_jacobian = 0.0;
        else:
            z = z + tf.expand_dims(self.b, 1);
            log_det_jacobian = 0.0;
        sum_log_det_jacobians += log_det_jacobian;
        return z, sum_log_det_jacobians;

class ElemMultLayer(Layer):
    def __init__(self, name, dim):
        self.name = name;
        self.dim = dim;
        self.param_names = ['a'];
        self.lock = False;
        
    def get_layer_info(self,):
        a_dim = (self.dim,1);
        dims = [a_dim];
        initializers = [tf.glorot_uniform_initializer()];
        return self.name, self.param_names, dims, initializers, self.lock;

    def connect_parameter_network(self, theta_layer):
        self.a = theta_layer[0];
        # params will have batch dimension if result of parameter network
        self.param_network = (len(self.a.shape) == 3);
        return None;
            
    def forward_and_jacobian(self, z, sum_log_det_jacobians):
        K, M, D, T = efn_tensor_shape(z);
        print('here a');
        print(self.a.shape);
        log_det_fac = tf.cast(T, tf.float64)*tf.log(tf.abs(tf.reduce_prod(self.a)));
        log_det_jacobian = log_det_fac*tf.ones((K,M), dtype=tf.float64);
        if (not self.param_network):
            a = tf.expand_dims(tf.expand_dims(self.a, 0), 0);
            z = tf.multiply(z, a);
        else:
            z = tf.multiply(z, tf.expand_dims(self.a, 1));
        print(log_det_jacobian.shape);
        print(sum_log_det_jacobians.shape);
        sum_log_det_jacobians += log_det_jacobian; 
        return z, sum_log_det_jacobians;


# Latent dynamical flows

class GP_Layer(Layer):
    def __init__(self, name, dim, inits, lock):
        self.name = name;
        self.D = dim;
        self.tau_init = inits['tau_init'];
        self.lock = lock;
        self.param_names = ['log_tau'];
        
    def get_layer_info(self,):
        log_tau_dim = (self.D,);
        dims = [log_tau_dim];
        print('tau initialization:');
        print(self.tau_init);
        initializers = [tf.constant(np.log(self.tau_init))];
        return self.name, self.param_names, dims, initializers, self.lock;

    def connect_parameter_network(self, theta_layer):
        self.log_tau = theta_layer[0];
        # params will have batch dimension if result of parameter network
        self.param_network = (len(self.log_tau.shape) == 2);
        return None;
            
    def forward_and_jacobian(self, z, sum_log_det_jacobians, ts):
        # TODO: his isn't going to work in an EFN yet
        _K, M, D, T = efn_tensor_shape(z);
        ds = tf.expand_dims(ts, 1) - tf.expand_dims(ts, 0);
        eps = .00001;
        tau = tf.exp(self.log_tau);
        z_outs = [];
        log_det_jacobian = 0.0;
        for i in range(self.D):
            K = tf.exp(-(tf.square(ds) / (2*tf.square(tau[i]))));

            z_GP_Sigma = K + eps*tf.eye(T, dtype=tf.float64); 
            L = tf.cholesky(z_GP_Sigma);
            log_det_jacobian = log_det_jacobian + tf.log(tf.abs(tf.reduce_prod(tf.matrix_diag_part(L))))

            z_out_i = tf.expand_dims(tf.transpose(tf.tensordot(L, z[:,:,i,:], [1, 2]), [1,2,0]), 2);
            z_outs.append(z_out_i);

        z_out = tf.concat(z_outs, 2);
        #print(z_out.shape);
        sum_log_det_jacobians += log_det_jacobian;
        return z_out, sum_log_det_jacobians;


class AR_Layer(Layer):
    def __init__(self, name, dim, T, P, inits, lock):
        self.name = name;
        self.D = dim;
        self.T = T;
        self.P = P;
        self.param_init = inits;
        self.lock = lock;
        self.param_names = ['alpha', 'log_sigma'];
        
    def get_layer_info(self,):
        alpha_dim = (self.D,self.P);
        log_sigma_dim = (self.D,);
        dims = [alpha_dim, log_sigma_dim];
        print('alpha initialization:');
        alpha_init = self.param_init['alpha_init'];
        sigma_init = self.param_init['sigma_init'];
        initializers = [tf.constant(alpha_init), tf.constant(np.log(sigma_init))];
        return self.name, self.param_names, dims, initializers, self.lock;

    def connect_parameter_network(self, theta_layer):
        self.alpha = theta_layer[0];
        self.log_sigma = theta_layer[1];
        # params will have batch dimension if result of parameter network
        self.param_network = (len(self.alpha.shape) == 3);
        return None;
            
    def forward_and_jacobian(self, z, sum_log_det_jacobians):
        # TODO: his isn't going to work in an EFN yet
        _K, M, D, _T = efn_tensor_shape(z);
        eps = .00001;
        sigma = tf.exp(self.log_sigma);
        z_outs = [];
        log_det_jacobian = tf.zeros((_K ,M), dtype=tf.float64);
        for i in range(self.D):
            alpha_i = self.alpha[i];
            sigma_i = sigma[i];
            z_GP_Sigma = AR_to_autocov_tf(alpha_i, sigma_i, self.P, self.T);
            L = tf.cholesky(z_GP_Sigma);
            log_det_jacobian = log_det_jacobian + tf.log(tf.abs(tf.reduce_prod(tf.matrix_diag_part(L))))

            z_out_i = tf.expand_dims(tf.transpose(tf.tensordot(L, z[:,:,i,:], [1, 2]), [1,2,0]), 2);
            z_outs.append(z_out_i);

        z_out = tf.concat(z_outs, 2);
        #print(z_out.shape);
        sum_log_det_jacobians += log_det_jacobian;
        return z_out, sum_log_det_jacobians;


# This layer class needs some attention
class VAR_Layer(Layer):
    def __init__(self, name, dim, T, P, inits, lock):
        self.name = name;
        self.D = dim;
        self.T = T;
        self.P = P;
        self.param_init = param_init;
        self.lock = lock;
        self.param_names = ['A', 'log_sigma'];
        
    def get_layer_info(self,):
        A_dim = (self.P, self.D, self.D);
        log_sigma_dim = (self.D,);
        dims = [A_dim, log_sigma_dim];
        print('A initialization:');
        As_init = self.param_init['A_init'];
        sigma_init = self.param_init['sigma_init'];
        initializers = [tf.constant(As_init), tf.constant(np.log(sigma_init))];
        return self.name, self.param_names, dims, initializers, self.lock;

    def connect_parameter_network(self, theta_layer):
        self.A = theta_layer[0];
        self.log_sigma = theta_layer[1];
        # params will have batch dimension if result of parameter network
        self.param_network = (len(self.A.shape) == 4);
        return None;
            
    def forward_and_jacobian(self, z, sum_log_det_jacobians):
        # TODO: his isn't going to work in an EFN yet
        _K, M, D, _T = efn_tensor_shape(z);
        eps = .00001;
        sigma = tf.exp(self.log_sigma);
        z_outs = [];
        log_det_jacobian = tf.zeros((_K ,M), dtype=tf.float64);
    
        z_GP_Sigma = compute_VAR_cov_tf(self.A, sigma, D, self.T, self.P);

        L = tf.cholesky(z_GP_Sigma);
        print('L');
        print(L.shape);

        print(log_det_jacobian.shape);
        log_det_jacobian = log_det_jacobian + tf.log(tf.abs(tf.reduce_prod(tf.matrix_diag_part(L))))
        print(log_det_jacobian.shape);
        z = tf.reshape(z, [_K, M, D*_T, 1]);
        L = tf.expand_dims(tf.expand_dims(L, 0), 0);
        L = tf.tile(L, [_K, M, 1, 1]);
        z_out = tf.matmul(L, z);
        z_out = tf.transpose(tf.reshape(z_out, [_K, M, _T, D]), [0,1,3,2]);
        
        #print(z_out.shape);
        sum_log_det_jacobians += log_det_jacobian;
        return z_out, sum_log_det_jacobians;


class GP_EP_CondRegLayer(Layer):
    def __init__(self, name, dim, T, Tps, tau_init, lock):
        self.name = name;
        self.D = dim;
        self.Tps = Tps;
        self.C = len(Tps);
        self.tau_init = tau_init;
        self.lock = lock;
        self.param_names = ['log_tau'];
        
    def get_layer_info(self,):
        log_tau_dim = (self.D,);
        dims = [log_tau_dim];
        print('tau initialization:');
        print(self.tau_init);
        initializers = [tf.constant(np.log(self.tau_init))];
        return self.name, self.param_names, dims, initializers, self.lock;

    def connect_parameter_network(self, theta_layer):
        self.log_tau = theta_layer[0];
        # params will have batch dimension if result of parameter network
        self.param_network = (len(self.log_tau.shape) == 2);
        return None;
            
    def forward_and_jacobian(self, z, sum_log_det_jacobians, ts):
        # TODO: this isn't going to work in an EFN yet
        _K, M, D, T = efn_tensor_shape(z);
        eps = 0.0;

        tau = tf.exp(self.log_tau);
        ts_0 = ts[0];
        middle_len = tf.shape(ts_0)[0];

        # get endpoints
        z_first = tf.expand_dims(z[:,:,:,0], 3);
        z_last = tf.expand_dims(z[:,:,:,1], 3);
        z_EP = tf.concat((z_first, z_last), 3);


        z_outs = [];
        log_det_jacobian = 0.0;
        for i in range(self.D):
            z_i = z[:,:,i,:];
            z_EP_i = z_EP[:,:,i,:];
            t_ind = 2;
            z_out_is = [];
            for j in range(self.C):
                Tp_j = self.Tps[j];
                ts = tf.concat([tf.constant(np.array([0.0]), dtype=tf.float64), tf.constant(np.array([Tp_j]), tf.float64), ts_0], axis=0);
                ds = tf.expand_dims(ts, 1) - tf.expand_dims(ts, 0);
                K = tf.exp(-(tf.square(ds) / (2*tf.square(tau[i]))));

                #z_middle = tf.slice(z_i, [0, 0, t_ind], [_K, M, middle_len]);
                z_middle = z_i[:,:,t_ind:(t_ind+middle_len)];
                t_ind = t_ind + middle_len;
            
                EP_inds = [0,1];
                K_allEP = K[:,:2];
                K_EPEP = K_allEP[:2, :];
                K_EPEP_inv = tf.matrix_inverse(K_EPEP);
                K_predEP = K_allEP[2:, :];
                A_mu = tf.matmul(K_predEP, K_EPEP_inv);
                z_GP_mu = tf.transpose(tf.tensordot(A_mu, z_EP_i, [1, 2]), [1,2,0]);
                #print(z_GP_mu.shape);
                K_EPpred = tf.transpose(K_predEP);
                K_predpred = K[2:,2:];
                z_GP_Sigma = K_predpred - tf.matmul(K_predEP, tf.matmul(K_EPEP_inv, K_EPpred));
                z_GP_Sigma = z_GP_Sigma + eps*tf.eye(middle_len, dtype=tf.float64); 
                L = tf.cholesky(z_GP_Sigma);
                log_det_jacobian = log_det_jacobian + tf.log(tf.abs(tf.reduce_prod(tf.matrix_diag_part(L))))

                z_GP_hat = z_GP_mu + tf.transpose(tf.tensordot(L, z_middle, [1,2]), [1,2,0]);
                z_out_ij = tf.expand_dims(tf.concat((z_first[:,:,i,:], z_GP_hat, z_last[:,:,i,:]), 2), 2);
                z_out_is.append(z_out_ij);

            z_out_i = tf.concat(z_out_is, 2);
            z_outs.append(tf.expand_dims(z_out_i, 3));

        z_out = tf.concat(z_outs, 3); # will be K x M x C x D x T

        sum_log_det_jacobians += log_det_jacobian;
        return z_out, z[:,:,:,t_ind:], sum_log_det_jacobians;



class GP_EP_CondRegFillLayer(Layer):
    def __init__(self, name, dim, T, Tps):
        self.name = name;
        self.D = dim;
        self.Tps = Tps;
        self.C = len(Tps);
        self.param_names = [];
        self.lock = False;
        
    def get_layer_info(self,):
        dims = [];
        initializers = None;
        return self.name, self.param_names, dims, initializers, self.lock;

    def connect_parameter_network(self, log_tau):
        self.log_tau = log_tau;
        # params will have batch dimension if result of parameter network
        self.param_network = (len(self.log_tau.shape) == 2);
        return None;
            
    def forward_and_jacobian(self, z, z0, sum_log_det_jacobians, ts):
        # TODO: this isn't going to work in an EFN yet
        z_shape = tf.shape(z);
        _K = z_shape[0];
        M = z_shape[1];
        D = z_shape[2];
        C = z_shape[3];
        T0 = z_shape[4];
        eps = 0.0

        tau = tf.exp(self.log_tau);
        ts_0 = ts[0];

        ts_1 = ts[1];
        ts_preds = [ts_1];
        ts_i = ts_1;
        for i in range(2,len(ts)):
            ts_i = tf.concat((ts_i, ts[i]), 0);
            ts_preds.append(ts_i);

        print('ts_preds', ts_preds);
        z_outs = [];
        log_det_jacobian = 0.0;
        for d in range(self.D):
            z_d = z[:,:,d,:,:];
            t_ind = 0;
            z_out_ds = [z_d[:,:,0,:]];
            t_ind = 0;
            for j in range(1,self.C):
                Tp_j = self.Tps[j];
                ts_data = tf.concat([tf.constant(np.array([0.0]), dtype=tf.float64), ts_0, tf.constant(np.array([Tp_j]), tf.float64)], axis=0);
                ts_pred = ts_preds[j-1];
                T_pred_j = tf.shape(ts_pred)[0];
                ts_all = tf.concat((ts_data, ts_pred), 0);
                ds = tf.expand_dims(ts_all, 1) - tf.expand_dims(ts_all, 0);

                K = tf.exp(-(tf.square(ds) / (2*tf.square(tau[d]))));
            
                K_data = K[:T0,:T0];
                K_data_inv = tf.matrix_inverse(K_data);
                K_pred_data = K[T0:,:T0];
                A_mu = tf.matmul(K_pred_data, K_data_inv);
                z_GP_mu = tf.transpose(tf.tensordot(A_mu, z_d[:,:,j,:], [1, 2]), [1,2,0]);
                #print(z_GP_mu.shape);
                K_data_pred = tf.transpose(K_pred_data);
                K_pred = K[T0:,T0:];
                z_GP_Sigma = K_pred - tf.matmul(K_pred_data, tf.matmul(K_data_inv, K_data_pred));
                z_GP_Sigma = z_GP_Sigma + eps*tf.eye(T_pred_j, dtype=tf.float64); 
                L = tf.cholesky(z_GP_Sigma);
                log_det_jacobian = log_det_jacobian + tf.log(tf.abs(tf.reduce_prod(tf.matrix_diag_part(L))))

                z0_dj = z0[:,:,d,t_ind:(t_ind+T_pred_j)];
                Lz = tf.transpose(tf.tensordot(L, z0_dj, [1,2]), [1,2,0]);

                z_out_ds.append(z_d[:,:,j,:-1]);
                z_out_ds.append(z_GP_mu + Lz);
                z_out_ds.append(tf.expand_dims(z_d[:,:,j,-1], 2));
                t_ind = t_ind + T_pred_j;

            z_out_d = tf.concat(z_out_ds, 2);
            z_outs.append(tf.expand_dims(z_out_d, 2));

        z_out = tf.concat(z_outs, 2); # will be K x M x D x CT

        sum_log_det_jacobians += log_det_jacobian;
        return z_out, sum_log_det_jacobians;


# this may need some attention
def AR_to_autocov_tf(alpha, sigma_eps, P, T):
    print(T);
    print(T-1);
    print(P);
    print(T-1 < P);
    if (T-1 < P):
        P = T-1;
    if (P==1):
        gamma0 = (sigma_eps**2) / (1 - (alpha[0]**2));
        gamma1 = alpha[0]*gamma0;
        gammas = [gamma0, gamma1];
        for i in range(2,T):
            gammas.append(alpha[0]*gammas[i-1]);
        
    elif (P==2):
        gamma0 = (sigma_eps**2) / (1 - ((alpha[0]**2 + alpha[1]*alpha[0]**2) / (1 - alpha[1])) - (alpha[1]**2));
        gamma1 = (alpha[0]*gamma0) / (1 - alpha[1]);
        gamma2 = alpha[0]*gamma1 + alpha[1]*gamma0;
        gammas = [gamma0, gamma1, gamma2];
        for i in range(3, T):
            gammas.append(alpha[0]*gammas[i-1] + alpha[1]*gammas[i-2]);
            
    elif (P==3):
        beta = (alpha[0] + alpha[1]*alpha[2]) / ((1 - alpha[1])*(1-((alpha[2]*(alpha[0]+alpha[2])) / (1 - alpha[1]))));
        
        term1_denom = 1 - (alpha[1]**2) - (alpha[2]**2) - (alpha[0]*alpha[1]*alpha[2]);
        term1_denom = term1_denom - (alpha[0] + alpha[1]*(alpha[0]+alpha[2]) + alpha[1]*alpha[2] + (alpha[0]*alpha[2]*(alpha[0]+alpha[2])))*beta;
        
        gamma0 = (sigma_eps**2) / term1_denom;
        gamma1 = beta*gamma0;
        gamma2 = (alpha[0] + alpha[2])*gamma1 + alpha[1]*gamma0;
        gamma3 = alpha[0]*gamma2 + alpha[1]*gamma1 + alpha[2]*gamma0;
        
        gammas = [gamma0, gamma1, gamma2, gamma3];
        for i in range(4, T):
            gammas.append(alpha[0]*gammas[i-1] + alpha[1]*gammas[i-2] + alpha[2]*gammas[i-3]);
        
        
    else:
        raise NotImplementedError();
        
    gammas_rev = gammas[::-1]; # reverse list
        
    Sigma_rows = [];
    for i in range(T):
        first = gammas_rev[T-(i+1):(T-1)];
        second = gammas_rev[:(T-i)];
        row_i = gammas_rev[T-(i+1):(T-1)] + gammas[:(T-i)];
        Sigma_rows.append(row_i);

    Sigma = tf.convert_to_tensor(Sigma_rows);
    
    return Sigma;


def compute_VAR_cov_tf(As, Sigma_eps, D, T, K):
    print('******************');
    print('compute_VAR_cov_tf T=%d' % T);
    print(As.shape);
    print(Sigma_eps.shape);
    # initialize the covariance matrix
    zcov = [[tf.eye(D, dtype=tf.float64)]];

    # compute the lagged noise-state covariance
    gamma = [tf.diag(Sigma_eps)];
    for t in range(1,T):
        print('1', t);
        gamma_t = tf.zeros((D,D), dtype=tf.float64);
        for k in range(1, min(t,K)+1):
            gamma_t += tf.matmul(As[k-1], gamma[t-k]);
        gamma.append(gamma_t);

    # compute the off-block-diagonal covariance blocks
    # first row
    for s in range(1, T):
        print('2', s);
        zcov_0s = tf.zeros((D,D), dtype=tf.float64);
        for k in range(1, min(s, K)+1):
            tau = s-k;
            zcov_0tau = zcov[0][tau];
            zcov_0s += tf.matmul(zcov_0tau, tf.transpose(As[k-1]));
        zcov[0].append(zcov_0s);
        zcov.append([tf.transpose(zcov_0s)]);

    # remaining rows
    for t in range(1,T):
        print('3', t);
        for s in range(t, T):
            print('(t,s) = (%d,%d)' % (t,s));
            zcov_ts = tf.zeros((D,D), dtype=tf.float64);
            # compute the contribution of lagged state-state covariances
            for k_t in range(1, min(t,K)+1):
                tau_t = t-k_t;
                for k_s in range(1, min(s,K)+1):
                    tau_s = s-k_s;
                    zcov_tauttaus = zcov[tau_t][tau_s];
                    zcov_ts += tf.matmul(As[k_t-1], tf.matmul(zcov_tauttaus, tf.transpose(As[k_s-1])));
            # compute the contribution of lagged noise-state covariances
            if (t==s):
                zcov_ts += Sigma_eps;
            for k in range(1, min(s,K)+1):
                tau_s = s-k;
                if (tau_s >= t):
                    print('adding gamma');
                    zcov_ts += tf.matmul(tf.transpose(gamma[tau_s-t]), tf.transpose(As[k-1]));

            zcov[t].append(zcov_ts);
            if (t != s):
                zcov[s].append(tf.transpose(zcov_ts));
                
    zcov = tf.convert_to_tensor(zcov);
    Zcov = tf.reshape(tf.transpose(zcov, [0,2,1,3]), (D*T, D*T));
    return Zcov;
