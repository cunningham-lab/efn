import tensorflow as tf
p_eps = 1e-6;

# Define the planar flow
class Layer:
    def forward_and_jacobian(self, y):
        raise NotImplementedError(str(type(self)));
        
class PlanarFlowLayer(Layer):

    def __init__(self, name='PlanarFlow', dim=1):
        raise NotImplementedError(name);
        # TODO this entire class needs to be compatible with the parameter network
        self.name = name;
        self.D = dim;
        
    def getWeights(self, name, D):
        u = tf.get_variable(name+'_u', shape=[D,1], dtype=tf.float32, \
                            initializer=tf.glorot_uniform_initializer());
        w = tf.get_variable(name+'_w', shape=[D,1], dtype=tf.float32, \
                            initializer=tf.glorot_uniform_initializer());
        b = tf.get_variable(name+'_b', shape=[1,1], dtype=tf.float32, \
                            initializer=tf.glorot_uniform_initializer());
        return u, w, b;
            
    def forward_and_jacobian(self, z, sum_log_det_jacobians, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            u, w, b = self.getWeights('weights', self.D);
            # helper function phi_k(z_{k-1})
            # derivative of tanh(x) is (1-tanh^2(x))
            phi = tf.matmul(w, (1.0 - tf.square(tf.tanh(tf.matmul(tf.transpose(w),z) + b))));
            # compute the running sum of log-determinant of jacobians
            sum_log_det_jacobians += tf.log(p_eps+tf.abs(1.0 + tf.matmul(tf.transpose(u), phi)))
            # compute z for this layer
            z = z + tf.matmul(u, tf.tanh(tf.matmul(tf.transpose(w), z) + b));
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
        sum_log_det_jacobians += tf.log(p_eps+tf.abs(tf.matrix_determinant(A)))
        return z, sum_log_det_jacobians;





