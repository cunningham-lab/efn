import tensorflow as tf
import numpy as np

def trainNN(y, x, L, units_per_layer, activation=tf.nn.tanh, lr=1e-2, num_iters=1000):
	n = x.shape[0];
	n_test = n//2;
	n_train = n-n_test;
	x_train = x[:n_train,:];
	y_train = y[:n_train,:];
	x_test = x[n_train:,:];
	y_test = y[n_train:,:];
	TSS_train = np.sum(np.square(y_train - np.mean(y_train)));
	TSS_test = np.sum(np.square(y_test - np.mean(y_test)));

	tf.reset_default_graph();
	D_y = y.shape[1];
	D_x = x.shape[1];
	X = tf.placeholder(dtype=tf.float32, shape=(None, D_x));
	Y_targ = tf.placeholder(dtype=tf.float32, shape=(None, D_y));
	h = X;
	for i in range(L):	
		print('making layer %d' % (i+1));
		h = tf.layers.dense(h, units_per_layer, activation=activation);
	A = tf.get_variable('A', shape=(D_y, units_per_layer), dtype=tf.float32);
	b = tf.get_variable('b', shape=(D_y,1));

	Y = tf.matmul(h, tf.transpose(A)) + tf.transpose(b);
	RSS = tf.reduce_sum(tf.square(Y_targ-Y));
	optimizer = tf.train.AdamOptimizer(lr);
	train_step = optimizer.minimize(RSS);

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer());
		_RSS = sess.run(RSS, {X:x_train, Y_targ:y_train});
		print('pre training R2: %.3f' % (1.0-(_RSS/TSS_train)));
		for i in range(num_iters):
			_ts, _RSS_train = sess.run([train_step, RSS], {X:x_train, Y_targ:y_train});
			if (np.mod(i,100) == 0):
				train_R2 = 1.0 - (_RSS_train/TSS_train);
				_RSS_test = sess.run(RSS, {X:x_test, Y_targ:y_test});
				test_R2 = 1.0 - (_RSS_test/TSS_test);
				print('%d: train R2=%.3f, test R2=%.3f' % (i, train_R2, test_R2));
		print('finished');
	return train_R2, test_R2;
