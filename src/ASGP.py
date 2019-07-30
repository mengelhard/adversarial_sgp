import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from SGPModel import SGPModel
from MixtureModel import MixtureModel, SGPGenerator, kmeans_mixture_model


class ASGP:

    def __init__(self, sgp_gen, sgpm,
                 filter_dist=1e-5,
                 n0=1, mw_alpha=20.,
                 g_lr=1e-1, d_lr=1e-4):

        self.z = tf.reshape(
            sgp_gen.samples,
            shape=(-1, sgp_gen._num_z, sgp_gen.dim_x))

        sgpm.set_khps(sgp_gen.sls, sgp_gen.sfs, sgp_gen.noise)
        sgpm.set_inducing_inputs(self.z, filter_dist=filter_dist)

        self.sgp_gen = sgp_gen
        self.sgpm = sgpm

        self.g_lr = g_lr
        self.d_lr = d_lr

        self.nlog_prior = sgp_gen.nlog_prior(
            n0=n0, alpha=mw_alpha)

        self._set_adversary()
        self._set_dloss()
        self._set_gloss()
        self._set_train_steps()

    def _set_adversary(self):

        with tf.variable_scope('adversary'):
            with tf.variable_scope('ref_samples'):
                self.d_ref_samples = adversary(self.sgp_gen.ref_samples)
            with tf.variable_scope('gen_samples'):
                self.d_gen_samples = adversary(self.sgp_gen.samples)

    def _set_dloss(self):

        dloss_total = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.d_gen_samples, labels=tf.zeros_like(self.d_gen_samples))
        dloss_total += tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.d_ref_samples, labels=tf.ones_like(self.d_ref_samples))
        self.dloss = tf.reduce_mean(dloss_total, axis=0)

    def _set_gloss(self):

        # follows Li et al formulation

        rlgs = tf.reduce_sum(
            tf.boolean_mask(
                tf.reshape(
                    self.sgp_gen.ref_logprob_gen_samples,
                    shape=(-1, self.sgp_gen._num_z)),
                self.sgpm.mask),
            axis=1)

        dgs = tf.reduce_sum(
            tf.boolean_mask(
                tf.reshape(
                    self.d_gen_samples,
                    shape=(-1, self.sgp_gen._num_z)),
                self.sgpm.mask),
            axis=1)

        nlog_marglik = self.sgpm.nlog_marglik()

        self.gloss = tf.reduce_mean(
            rlgs + nlog_marglik - dgs, axis=0) + self.nlog_prior

    def _set_train_steps(self):

        dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'adversary')
        gvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

        self.dtrain_step = tf.train.AdamOptimizer(self.d_lr).minimize(
            self.dloss, var_list=dvars)
        self.gtrain_step = tf.train.AdamOptimizer(self.g_lr).minimize(
            self.gloss, var_list=gvars)


def learn_asgp(tx, ty, vx, vy, mx, my,
               num_ind_inputs=None,
               mw_alpha=None,
               num_sgp_samples=100,
               max_steps=5000,
               check_val_freq=10,
               max_no_improve=20,
               gaussian_reference=False,
               weights_nn_depth=2,
               g_lr=1e-1,
               d_lr=1e-4):

    tf.reset_default_graph()

    num_components = num_ind_inputs

    with tf.variable_scope('generator'):

        sgp_gen = SGPGenerator(
            tx, num_components, num_ind_inputs, num_sgp_samples,
            component_weights_gen_type='nn-gumbel-softmax',
            weights_nn_depth=weights_nn_depth,
            tau_initial=1e-4,
            gaussian_reference=gaussian_reference)

    sgpm = SGPModel(tx, ty, jitter_magnitude=1e-4)

    asgp = ASGP(
        sgp_gen, sgpm,
        mw_alpha=mw_alpha,
        n0=1.,
        filter_dist=1e-4,
        g_lr=g_lr,
        d_lr=d_lr)

    mask = sgpm.mask

    test_predictions = sgpm.predict(mx)
    val_predictions = sgpm.predict(vx)

    test_nmse = normalized_mean_square_error(
        tf.reduce_mean(test_predictions, axis=0), my)
    val_nmse = normalized_mean_square_error(
        tf.reduce_mean(val_predictions, axis=0), vy)

    with tf.Session() as s:

        s.run(tf.global_variables_initializer())

        for i in range(100):
            _ = s.run(asgp.dtrain_step)

        idx = 0
        no_improve = 0
        best = 100.

        gloss_all = []
        val_nmse_all = []

        while (no_improve < max_no_improve) and (idx < max_steps):

            idx += 1

            _ = s.run(asgp.dtrain_step)
            _, gloss_, mask_ = s.run(
                [asgp.gtrain_step, asgp.gloss, sgpm.mask])

            gloss_all.append(gloss_)

            prc_used = np.sum(mask_) / len(mask_)

            if idx % check_val_freq == 0:

                val_nmse_ = s.run(val_nmse)
                val_nmse_all.append(val_nmse_)
                print('Current validation NMSE: %.3f' % val_nmse_)
                print('Percent valid SGP samples: %.0f' % (100 * prc_used))

                if best < val_nmse_:
                    no_improve += 1
                else:
                    best = val_nmse_
                    no_improve = 0

        test_nmse_ = s.run(test_nmse)
        print('Final test NMSE: %.3f' % test_nmse_)

    return gloss_all, val_nmse_all, test_nmse_


def adversary(y):

    with slim.arg_scope([slim.fully_connected], activation_fn=lrelu):

        net = slim.fully_connected(y, 256, scope='fc_0')

        for i in range(5):
            dnet = slim.fully_connected(
                net, 256, scope='fc_%d_r0' % (i + 1))
            net += slim.fully_connected(
                dnet, 256, activation_fn=None, scope='fc_%d_r1' % (i + 1),
                weights_initializer=tf.constant_initializer(0.))
            net = lrelu(net)

    T = slim.fully_connected(
        net, 1, activation_fn=None, scope='T',
        weights_initializer=tf.constant_initializer(0.))
    T = tf.squeeze(T, [1])

    return T


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def normalized_mean_square_error(pred, real, return_mse=False):
    mse = mean_square_error(pred, real)
    if return_mse:
        return mse / mean_square_error(0, real), mse
    else:
        return mse / mean_square_error(0, real)


def mean_square_error(pred, real):
    real = tf.cast(real, dtype=tf.float32)
    se = (pred - real)**2
    return tf.reduce_mean(se, axis=0)
