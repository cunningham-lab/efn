import tensorflow as tf
p_eps = 1e-6;

# Define the planar flow
class Layer:
    def forward_and_jacobian(self, y):
        raise NotImplementedError(str(type(self)));
        
class PlanarFlowLayer(Layer):

    def __init__(self, name='PlanarFlow', dim=1):
        # TODO this entire class needs to be compatible with the parameter network
        self.name = name;
        self.dim = dim;
        self.param_names = ['u', 'w', 'b'];

    def get_layer_info(self,):
        u_dim = (self.dim,1);
        w_dim = (self.dim,1);
        b_dim = (1,1);
        dims = [u_dim, w_dim, b_dim];
        return self.name, self.param_names, dims;
        
    def connect_parameter_network(self, theta_layer):
        self.u = theta_layer[0];
        self.w = theta_layer[1];
        self.b = theta_layer[2];
        self.param_network = (len(self.u.shape) == 3);
        return None;
            
    def forward_and_jacobian(self, z, sum_log_det_jacobians, reuse=False):
        M = tf.shape(z)[1];
        if (not self.param_network):
            u = tf.expand_dims(tf.tile(tf.expand_dims(self.u, 0), [M, 1, 1]), 0);
            w = tf.expand_dims(tf.tile(tf.expand_dims(self.w, 0), [M, 1, 1]), 0);
            b = tf.expand_dims(tf.tile(tf.expand_dims(self.b, 0), [M, 1, 1]), 0);
        else:
            u = tf.tile(tf.expand_dims(self.u, 1), [1, M, 1, 1]);
            w = tf.tile(tf.expand_dims(self.w, 1), [1, M, 1, 1]);
            b = tf.tile(tf.expand_dims(self.b, 1), [1, M, 1, 1]);

        # helper function phi_k(z_{k-1})
        # derivative of tanh(x) is (1-tanh^2(x))
        phi = tf.matmul(w, (1.0 - tf.square(tf.tanh(tf.matmul(tf.transpose(w, [0,1,3,2]), z) + b))));
        # compute the running sum of log-determinant of jacobians
        log_det_jacobian = tf.log(tf.abs(1.0 + tf.matmul(tf.transpose(u, [0,1,3,2]), phi)));
        sum_log_det_jacobians += log_det_jacobian[:,:,0,0];
        # compute z for this layer
        nonlin_term = tf.tanh(tf.matmul(tf.transpose(w, [0,1,3,2]), z) + b);
        z = z + tf.matmul(u, nonlin_term);
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
        if (not self.param_network):
            K = tf.shape(z)[0];
            M = tf.shape(z)[1];
            batch_size = tf.multiply(K,M);
            A = self.A;
            b = tf.expand_dims(tf.expand_dims(self.b, 0), 0);
            z = tf.tensordot(A, z, [[1], [2]]);
            z = tf.transpose(z, [1, 2, 0, 3]) + b;
            log_det_jacobian = tf.multiply(tf.log(p_eps + tf.abs(tf.matrix_determinant(A))), tf.ones((batch_size,)));
        else:
            M = tf.shape(z)[1];
            A = tf.tile(tf.expand_dims(self.A, 1), [1, M, 1, 1]);
            b = tf.tile(tf.expand_dims(self.b, 1), [1, M, 1, 1]);
            z = tf.matmul(A, z) + b;
            log_det_jacobian = tf.log(p_eps+tf.abs(tf.matrix_determinant(A)));
        sum_log_det_jacobians += log_det_jacobian;
        return z, sum_log_det_jacobians;





