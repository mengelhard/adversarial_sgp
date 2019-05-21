import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim


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


def learn_asgp(tx, ty):

    num_components = 4
    num_ind_inputs = 3
    num_sgp_samples = 100
    num_steps = 300

    with tf.variable_scope('generator'):

        sgp_gen = SGPGenerator(
            tx, num_components, num_ind_inputs, num_sgp_samples,
            component_weights_gen_type='nn-gumbel-softmax',
            weights_nn_depth=4,
            tau_initial=1e-4,
            gaussian_reference=True)

    sgpm = SGPModel(tx, ty, jitter_magnitude=1e-5)

    asgp = ASGP(
        sgp_gen, sgpm,
        mw_alpha=100.,
        n0=1.,
        filter_dist=1e-4,
        g_lr=1e-1)

    mask = sgpm.mask

    x_grid = np.linspace(0, 60, 200)
    predictions = sgpm.predict(x_grid[:, np.newaxis])

    with tf.Session() as s:

        s.run(tf.global_variables_initializer())

        initial_z_, ref_samples_ = s.run([sgp_gen.samples, sgp_gen.ref_samples])

        for i in range(100):

            _ = s.run(asgp.dtrain_step)

        progress = tqdm(range(num_steps))

        # plot_mcycle_initial(tx, ty, ref_samples_, initial_z_)

        for i in progress:

            _ = s.run(asgp.dtrain_step)
            _, gloss_, mask_ = s.run(
                [asgp.gtrain_step, asgp.gloss, sgpm.mask])

            prc_used = np.sum(mask_) / len(mask_)

            progress.set_description('gloss=%.3f, prc_used=%.3f' % (
                gloss_, prc_used))

        z_, ref_samples_, predictions_ = s.run(
            [sgp_gen.samples, sgp_gen.ref_samples, predictions])

    # plot_mcycle_final(tx, ty, ref_samples_, z_, x_grid, predictions_)


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
