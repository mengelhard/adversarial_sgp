import numpy as np
import tensorflow as tf


class SGPModel:

    def __init__(self, x, y, approx_method='vfe',
                 jitter_magnitude=1e-5):

        assert x.ndim == 2
        assert y.ndim == 1

        self.num_x, self.dim_x = np.shape(x)
        self.num_y = len(y)

        assert self.num_x == self.num_y

        self.x = tf.constant(x, dtype=tf.float32)[tf.newaxis, :, :]
        self.y = tf.constant(y, dtype=tf.float32)[tf.newaxis, :]

        self.approx_method = approx_method
        self.logtp = tf.constant(
            np.log(2. * np.pi),
            dtype=tf.float32)

        if approx_method == 'vfe':

            self.nlog_marglik = self._vfe_nlog_marglik
            self._set_terms = self._vfe_set_terms

        elif approx_method == 'fitc':

            self.nlog_marglik = self._fitc_nlog_marglik
            self._set_terms = self._fitc_set_terms

        self.jitter_magnitude = jitter_magnitude

    def set_kernel(self, sls, sfs, kernel_type='rbf'):

        self.batch_size = tf.shape(sfs)[0]
        self.sfs = sfs
        self.sls = sls

        if kernel_type == 'rbf':
            self.kernel = lambda x1, x2: rbf_kernel(x1, x2, sls, sfs)

    def set_noise(self, noise):

        self.noise = noise

    def set_inducing_inputs(self, z):

        self.z = z
        self.num_z = tf.shape(z)[1]
        self.kzz = self.kernel(z, z)
        self.kxz = self.kernel(self.x, z)
        self.kzx = tf.matrix_transpose(self.kxz)
        self.jitter = self.jitter_magnitude * tf.eye(
            self.num_z, dtype=tf.float32)[tf.newaxis, :, :]
        self.kzz_inverse = tf.matrix_inverse(self.kzz + self.jitter)
        self._qmm_diag = tf.reduce_sum(tf.matmul(
            self.kxz, self.kzz_inverse) * self.kxz, axis=2)
        self._set_terms()

    def fullgp_nlog_marglik(self):

        return None

    def predict(self, t):

        ktz = self._check_points_and_get_kernel(t, self.z)
        qm = self.kzz + self._kzx_gi_kxz + self.jitter
        ph = right_mult_by_vector(tf.matrix_inverse(qm), self._kgiy)
        return right_mult_by_vector(ktz, ph)

    def fullgp_predict(self, t):

        ktx = self._check_points_and_get_kernel(t, self.x)
        kxx = self.kernel(self.x, self.x)
        inner = kxx + self.noise[:, tf.newaxis, tf.newaxis] * tf.eye(
            self.num_x)[tf.newaxis, :, :]
        ph = right_mult_by_vector(tf.matrix_inverse(inner), self.y)
        return right_mult_by_vector(ktx, ph)

    def _check_points_and_get_kernel(self, t, u):

        assert t.ndim == 2
        assert np.shape(t)[1] == self.dim_x

        t = tf.constant(t, dtype=tf.float32)[tf.newaxis, :, :]
        return self.kernel(t, u)

    def _vfe_set_terms(self):

        self._gid = 1 / self.noise[:, tf.newaxis]
        self._giy = self._gid * self.y
        self._kgiy = right_mult_by_vector(self.kzx, self._giy)
        self._trace = self.sfs * self.num_x - tf.reduce_sum(
            self._qmm_diag, axis=1)
        self._kzx_gi_kxz = tf.matmul(
            self.kzx, self.kxz) / self.noise[:, tf.newaxis, tf.newaxis]

    def _fitc_set_terms(self):

        self._gd = (self.sfs + self.noise)[:, tf.newaxis] - self._qmm_diag
        self._gid = 1 / self._gd
        self._giy = self._gid * self.y
        self._kgiy = right_mult_by_vector(self.kzx, self._giy)
        self._trace = 0
        self._kzx_gi_kxz = kzx_gi_kxz = tf.matmul(
            right_mult_by_diag(self.kzx, self._gid), self.kxz)

    def _vfe_nlog_marglik(self):

        inner = self.kzz_inverse + self._kzx_gi_kxz + self.jitter

        covd = logdet(inner) + logdet(self.kzz + self.jitter)
        covd += tf.log(self.noise) * self.num_x

        t1 = .5 * self.num_x * self.logtp
        t2 = .5 * tf.reduce_sum(self.y * self.y * self._gid, axis=1)
        t2 -= .5 * tf.reduce_sum(tf.reduce_sum(
            self._kgiy[:, :, tf.newaxis] * tf.matrix_inverse(inner),
            axis=1) * self._kgiy, axis=1)
        t3 = .5 * covd
        t4 = .5 * tf.div(self._trace, self.noise)

        return t1 + t2 + t3 + t4

    def _fitc_nlog_marglik(self):

        inner = self.kzz_inverse + self._kzx_gi_kxz + self.jitter

        covd = logdet(inner) + logdet(self.kzz + self.jitter)
        covd += tf.reduce_sum(tf.log(self._gd), axis=1)

        t1 = .5 * self.num_x * self.logtp
        t2 = .5 * tf.reduce_sum(self.y * self.y * self._gid, axis=1)
        t2 -= .5 * tf.reduce_sum(tf.reduce_sum(
            self._kgiy[:, :, tf.newaxis] * tf.matrix_inverse(inner),
            axis=1) * self._kgiy, axis=1)
        t3 = .5 * covd

        return t1 + t2 + t3


def logdet(matrix):
    chol = tf.cholesky(matrix)
    return 2.0 * tf.reduce_sum(tf.log(tf.real(tf.matrix_diag_part(chol))),
                               reduction_indices=[-1])


def right_mult_by_diag(mat, diag, naxes=3):
    return mat * tf.expand_dims(diag, axis=naxes - 2)


def right_mult_by_vector(mat, vec):
    return tf.reduce_sum(right_mult_by_diag(mat, vec), axis=2)


def pairwise_distance(x1, x2):
    r1 = tf.reduce_sum(x1**2, axis=2)[:, :, tf.newaxis]
    r2 = tf.reduce_sum(x2**2, axis=2)[:, tf.newaxis, :]
    r12 = tf.matmul(x1, tf.matrix_transpose(x2))
    return r1 + r2 - 2 * r12


def scaled_square_dist(x1, x2, sls):
    '''sls is a P by D matrix of squared length scales,
    where P is the number of particles and D is the number of dimenions'''
    ls = tf.sqrt(sls)[:, tf.newaxis, :]
    return pairwise_distance(x1 / ls, x2 / ls)


def rbf_kernel(x1, x2, sls, sfs):
    ssd = scaled_square_dist(x1, x2, sls)
    return sfs[:, tf.newaxis, tf.newaxis] * tf.exp(-.5 * ssd)
