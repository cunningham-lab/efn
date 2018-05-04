import tensorflow as tf
import numpy as np
import scipy.linalg

def efn_tensor_shape(z):
    # could use eager execution to iterate over shape instead.  Avoiding for now.
    K = tf.shape(z)[0];
    M = tf.shape(z)[1];
    D = tf.shape(z)[2];
    T = tf.shape(z)[3];
    return K, M, D, T;

# Define the planar flow
class Layer:
    def __init__(self, name='SimplexBijectionLayer'):
        # TODO this entire class needs to be compatible with the parameter network
        self.name = name;
        self.param_names = [];
        self.param_network = False;

    def get_layer_info(self,):
        return self.name, [], [];

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

    def get_layer_info(self,):
        u_dim = (self.dim,1);
        w_dim = (self.dim,1);
        b_dim = (1,1);
        dims = [u_dim, w_dim, b_dim];
        return self.name, self.param_names, dims;

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
        log_det_jacobian = tf.reshape(log_det_jacobian, [K,M]); # update for more time samples

        # compute z for this layer
        nonlin_term = tf.tanh(tf.matmul(tf.transpose(w, [0,2,1]), z_KD_MTvec) + b);
        z = z + tf.transpose(tf.reshape(tf.matmul(u, nonlin_term), [K,D,M,T]), [0,2,1,3]);

        sum_log_det_jacobians += log_det_jacobian;
        return z, sum_log_det_jacobians;

class SimplexBijectionLayer(Layer):
    def __init__(self, name='SimplexBijectionLayer'):
        # TODO this entire class needs to be compatible with the parameter network
        self.name = name;
        self.param_names = [];
        self.param_network = False;

    def forward_and_jacobian(self, z, sum_log_det_jacobians):
        D = tf.shape(z)[2];
        ex = tf.exp(z);
        den = tf.reduce_sum(ex, 2) + 1.0;
        log_dets = tf.log(1.0 - (tf.reduce_sum(ex, 2) / den)) - tf.cast(D, tf.float64)*tf.log(den) + tf.reduce_sum(z, 2);
        z = tf.concat((ex / tf.expand_dims(den, 2), 1.0 / tf.expand_dims(den, 2)), axis=2);
        sum_log_det_jacobians += log_dets[:,:,0];
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
    def __init__(self, name='SimplexBijectionLayer'):
        # TODO this entire class needs to be compatible with the parameter network
        self.name = name;
        self.param_names = [];
        self.param_network = False;

    def forward_and_jacobian(self, z, sum_log_det_jacobians):
        z_out = tf.tanh(z);
        log_det_jacobian = tf.reduce_sum(tf.log(1.0 - (z_out**2)), 2);
        sum_log_det_jacobians += log_det_jacobian[:,:,0];
        return z, sum_log_det_jacobians;

class StructuredSpinnerLayer(Layer):
    def __init__(self, name, dim):
        self.name = name;
        self.dim = dim;
        self.param_names = ['d1', 'd2', 'd3', 'b'];
        self.d1 = None;
        self.d2 = None;
        self.d3 = None;
        self.b = None;
        
    def get_layer_info(self,):
        d1_dim = (self.dim, 1);
        d2_dim = (self.dim, 1);
        d3_dim = (self.dim, 1);
        b_dim = (self.dim,1);
        dims = [d1_dim, d2_dim, d3_dim, b_dim];
        return self.name, self.param_names, dims;

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
        #A = tf.matmul(H, tf.matmul(D2, tf.matmul(H, D1)));

        if (not self.param_network):
            b = tf.expand_dims(tf.expand_dims(b, 0), 0);
            z = tf.tensordot(A, z, [[1], [2]]);
            z = tf.transpose(z, [1, 2, 0, 3]) + b;
            log_det_A = tf.log(tf.abs(tf.reduce_prod(d1))) + tf.log(tf.abs(tf.reduce_prod(d2))) + tf.log(tf.abs(tf.reduce_prod(d3)));
            log_det_jacobian = tf.multiply(log_det_A, tf.ones((K*M,), dtype=tf.float64));
        else:
            z_KD_MTvec = tf.reshape(tf.transpose(z, [0,2,1,3]), [K,D,M*T]);
            Az_KD_MTvec = tf.matmul(A, z_KD_MTvec);
            Az = tf.transpose(tf.reshape(Az_KD_MTvec, [K,D,M,T]), [0,2,1,3]);
            z = Az + tf.expand_dims(b, 1);
            log_det_jacobian = tf.log(tf.abs(tf.matrix_determinant(A)));
            log_det_jacobian = tf.tile(tf.expand_dims(log_det_jacobian, 1), [1, M]);
        sum_log_det_jacobians += log_det_jacobian;
        return z, sum_log_det_jacobians;



class LinearFlowLayer(Layer):
    def __init__(self, name, dim):
        self.name = name;
        self.dim = dim;
        self.param_names = ['A', 'b'];
        
    def get_layer_info(self,):
        A_dim = (self.dim, self.dim);
        b_dim = (self.dim,1);
        dims = [A_dim, b_dim];
        return self.name, self.param_names, dims;

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
            log_det_jacobian = tf.multiply(tf.log(tf.abs(tf.matrix_determinant(A))), tf.ones((K*M,), dtype=tf.float64));
        else:
            z_KD_MTvec = tf.reshape(tf.transpose(z, [0,2,1,3]), [K,D,M*T]);
            Az_KD_MTvec = tf.matmul(self.A, z_KD_MTvec);
            Az = tf.transpose(tf.reshape(Az_KD_MTvec, [K,D,M,T]), [0,2,1,3]);
            z = Az + tf.expand_dims(self.b, 1);
            log_det_jacobian = tf.log(tf.abs(tf.matrix_determinant(self.A)));
            log_det_jacobian = tf.tile(tf.expand_dims(log_det_jacobian, 1), [1, M]);
        sum_log_det_jacobians += log_det_jacobian;
        return z, sum_log_det_jacobians;





