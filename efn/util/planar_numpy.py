import numpy as np

def planar_flow(z, u, w, b):
	D = z.shape[0];
	n = z.shape[1];
	add_term = np.dot(u, np.tanh(np.dot(w.T, z) + b));
	y = z + add_term;
	hprime = (1 - np.square(np.tanh(np.dot(w.T, z) + b)))
	psi = np.dot(w, hprime);
	log_det_jac1 = np.zeros((n,));
	log_det_jac2 = np.zeros((n,));
	for i in range(n):
		log_det_jac1[i] = np.log(np.abs(np.linalg.det(np.eye(D) + np.dot(u, np.expand_dims(psi[:,i], 0)))));
		log_det_jac2[i] = np.log(np.abs(1 + np.dot(u.T, np.expand_dims(psi[:,i], 1))));
	return y, log_det_jac1;


def to_simplex(z):
	D = z.shape[0];
	n = z.shape[1];
	y = np.exp(z) / np.expand_dims(np.sum(np.exp(z), 0) + 1, 0);
	det_jac = np.multiply(np.prod(y, 0), (1 - np.sum(y, 0)));
	log_det_jac1 = np.log(np.abs(det_jac));

	ex = np.exp(z);
	den = np.sum(ex, 0) + 1.0;
	log_det_jac2 = np.log(1 - (np.sum(ex, 0) / den)) - D*np.log(den) + np.sum(z, 0);
	y = np.concatenate((y, np.expand_dims(1-np.sum(y, 0), 0)), axis=0);
	return y, log_det_jac1, log_det_jac2;