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
        if (not self.param_network):
            batch_size = tf.shape(z)[0];
            u = tf.tile(tf.expand_dims(self.u, 0), [batch_size, 1, 1]);
            w = tf.tile(tf.expand_dims(self.w, 0), [batch_size, 1, 1]);
            b = tf.tile(tf.expand_dims(self.b, 0), [batch_size, 1, 1]);
        else:
            u = self.u;
            w = self.w;
            b = self.b;

        # helper function phi_k(z_{k-1})
        # derivative of tanh(x) is (1-tanh^2(x))
        phi = tf.matmul(w, (1.0 - tf.square(tf.tanh(tf.matmul(tf.transpose(w, [0,2,1]), z) + b))));
        # compute the running sum of log-determinant of jacobians
        log_det_jacobian = tf.log(p_eps+tf.abs(1.0 + tf.matmul(tf.transpose(u, [0,2,1]), phi)));
        sum_log_det_jacobians += log_det_jacobian;
        # compute z for this layer
        nonlin_term = tf.tanh(tf.matmul(tf.transpose(w, [0,2,1]), z) + b);
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
            batch_size = tf.shape(z)[0];
            A = tf.tile(tf.expand_dims(self.A, 0), [batch_size, 1, 1]);
            b = tf.tile(tf.expand_dims(self.b, 0), [batch_size, 1, 1]);
        else:
            A = self.A;
            b = self.b;
        z = tf.matmul(A, z) + b;
        T = tf.shape(z)[2];
        log_det_jacobian = tf.log(p_eps+tf.abs(tf.matrix_determinant(A)));
        log_det_jacobian = tf.expand_dims(tf.expand_dims(log_det_jacobian, 1), 2);
        log_det_jacobian = tf.tile(log_det_jacobian, [1, 1, T]);
        sum_log_det_jacobians += log_det_jacobian;
        return z, sum_log_det_jacobians;





