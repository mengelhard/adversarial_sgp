import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.contrib import slim
from tensorflow.contrib import distributions
import matplotlib.pyplot as plt
from load_data import load_dataset
from sgp import SGPModel
from MixtureModel import MixtureModel, SGPGenerator, kmeans_mixture_model
from learn_with_lr import ASGP


def main():

    tx, ty, vx, vy, mx, my = load_dataset('mcycle', val_prc=.1)

    # noise_, sfs_, sls_ = get_khps(tx, ty)

    # print('(from Full GP -- Noise: ', noise_, ', SFS: ', sfs_, ' SLS: ', sls_)

    # z_ = np.linspace(-10, 70, 100)

    # check_approx_marglik(
    #     tx, ty, noise_, sfs_, sls_, z_,
    #     savefile='../img/mcycle_experiments/marglik.png')

    # sample_from_initial_mm(
    #     tx, ty,
    #     savefile='../img/mcycle_experiments/initial_mm_samples.png')

    # learn_from_u(tx, ty, noise_, sfs_, sls_)

    learn_asgp(tx, ty)


def learn_asgp(tx, ty):

    num_components = 4
    num_ind_inputs = 4
    num_sgp_samples = 100
    num_steps = 300

    with tf.variable_scope('generator'):

        sgp_gen = SGPGenerator(
            tx, num_components, num_ind_inputs, num_sgp_samples,
            component_weights_gen_type='nn-gumbel-softmax',
            weights_nn_depth=5,
            tau_initial=1e-4,
            gaussian_reference=True)

    sgpm = SGPModel(tx, ty, jitter_magnitude=1e-4)

    asgp = ASGP(
        sgp_gen, sgpm,
        mw_alpha=200.,
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

        plot_mcycle_initial(tx, ty, ref_samples_, initial_z_)

        for i in progress:

            _ = s.run(asgp.dtrain_step)
            _, gloss_, mask_ = s.run(
                [asgp.gtrain_step, asgp.gloss, sgpm.mask])

            prc_used = np.sum(mask_) / len(mask_)

            progress.set_description('gloss=%.3f, prc_used=%.3f' % (
                gloss_, prc_used))

        z_, ref_samples_, predictions_ = s.run(
            [sgp_gen.samples, sgp_gen.ref_samples, predictions])

    plot_mcycle_final(tx, ty, ref_samples_, z_, x_grid, predictions_)


def plot_mcycle_initial(tx, ty, ref_samples, z):

    fig, ax = plt.subplots(ncols=1, nrows=3, sharex=True, figsize=(6, 6))

    ax[0].scatter(tx, ty)
    ax[0].set_ylabel('data (y)')
    ax[0].set_xlabel('data (x)')
    ax[1].hist(ref_samples, bins=np.linspace(0, 60, 30))
    ax[1].set_ylabel('num samples')
    ax[1].set_xlabel('reference samples')
    ax[2].hist(z, bins=np.linspace(0, 60, 30))
    ax[2].set_ylabel('num samples')
    ax[2].set_xlabel('initial z values')
    ax[2].set_xlim([0, 60])

    plt.tight_layout()
    plt.savefig('../img/mcycle_experiments/initial_samples')


def plot_mcycle_final(tx, ty, ref_samples, z, pred_x, pred_y):

    fig, ax = plt.subplots(ncols=1, nrows=4, sharex=True, figsize=(6, 8))

    ax[0].scatter(tx, ty)
    ax[0].set_ylabel('data (y)')
    ax[0].set_xlabel('data (x)')
    ax[1].hist(ref_samples, bins=np.linspace(0, 60, 30))
    ax[1].set_ylabel('number of samples')
    ax[1].set_xlabel('reference samples')
    ax[2].hist(z, bins=np.linspace(0, 60, 30))
    ax[2].set_ylabel('number of samples')
    ax[2].set_xlabel('final z values')

    mn = np.median(pred_y, axis=0)
    hi = np.percentile(pred_y, 75, axis=0)
    lo = np.percentile(pred_y, 25, axis=0)

    ax[3].scatter(tx, ty)
    ax[3].plot(pred_x, mn, color='m', alpha=.5)
    ax[3].fill_between(pred_x, lo, hi, color='m', alpha=.2)
    ax[3].set_ylabel('predict')
    ax[3].set_xlim([0, 60])

    plt.tight_layout()
    plt.savefig('../img/mcycle_experiments/final_samples.png')


def learn_from_u(tx, ty, noise_, sfs_, sls_,
                 num_components=4, batch_size=200,
                 num_ind_inputs=4, num_steps=1000,
                 tol_num_iter_no_improve=30):

    with tf.variable_scope('generator'):

        mm = MixtureModel(
            tx, num_components, batch_size,
            tau_initial=1e-4)

        z = mm.samples

        means = mm.means
        widths = mm.widths
        mixture_weights = mm.mixture_weights

    noise = tf.constant(
        noise_ * np.ones(batch_size // num_ind_inputs),
        dtype=tf.float32)
    sfs = tf.constant(
        sfs_ * np.ones(batch_size // num_ind_inputs),
        dtype=tf.float32)
    sls = tf.constant(
        sls_ * np.ones((batch_size // num_ind_inputs, 1)),
        dtype=tf.float32)

    z_reshaped = tf.reshape(z, shape=(-1, num_ind_inputs, 1))

    sgpm = SGPModel(tx, ty, jitter_magnitude=1e-5)
    sgpm.set_khps(sls, sfs, noise)
    sgpm.set_inducing_inputs(z_reshaped, filter_dist=1e-3)
    mask = sgpm.mask

    nlog_marglik = sgpm.nlog_marglik()
    nlog_prior = mm.nlog_prior(n0=1, alpha=5.)

    ref_logprob_gen_samples = mm.ref_logprob_gen_samples
    ref_samples = mm.ref_samples

    with tf.variable_scope('adversary'):
        with tf.variable_scope('ref_samples'):
            d_ref_samples = adversary(ref_samples)
        with tf.variable_scope('gen_samples'):
            d_gen_samples = adversary(z)

    dloss_total = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_gen_samples, labels=tf.zeros_like(d_gen_samples))
    dloss_total += tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_ref_samples, labels=tf.ones_like(d_ref_samples))
    dloss = tf.reduce_mean(dloss_total, axis=0)

    # follows Li et al formulation

    rlgs = tf.reduce_sum(
        tf.boolean_mask(
            tf.reshape(
                ref_logprob_gen_samples,
                shape=(-1, num_ind_inputs)),
            mask),
        axis=1)

    dgs = tf.reduce_sum(
        tf.boolean_mask(
            tf.reshape(
                d_gen_samples,
                shape=(-1, num_ind_inputs)),
            mask),
        axis=1)

    gloss = tf.reduce_mean(rlgs + nlog_marglik - dgs, axis=0) + nlog_prior
    # gloss = tf.reduce_mean(nlog_marglik, axis=0) + nlog_prior

    dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'adversary')
    gvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

    dtrain_step = tf.train.AdamOptimizer(1e-4).minimize(dloss, var_list=dvars)
    gtrain_step = tf.train.AdamOptimizer(1e-1).minimize(gloss, var_list=gvars)

    x_grid = np.linspace(0, 60, 200)
    predictions = sgpm.predict(x_grid[:, np.newaxis])

    best_gloss = np.inf
    all_gloss = []
    num_iter_no_improve = 0

    with tf.Session() as s:

        s.run(tf.global_variables_initializer())

        initial_z_, ref_samples_ = s.run([z, ref_samples])

        print('Z shape', np.shape(initial_z_))
        print('Ref samples shape', np.shape(ref_samples_))

        for i in range(100):

            _ = s.run(dtrain_step)

        progress = tqdm(range(num_steps))

        fig, ax = plt.subplots(ncols=1, nrows=3, sharex=True, figsize=(6, 6))

        ax[0].scatter(tx, ty)
        ax[0].set_ylabel('data (y)')
        ax[0].set_xlabel('data (x)')
        ax[1].hist(ref_samples_, bins=20)
        ax[1].set_ylabel('num samples')
        ax[1].set_xlabel('reference samples')
        ax[2].hist(initial_z_, bins=20)
        ax[2].set_ylabel('num samples')
        ax[2].set_xlabel('initial z values')

        plt.tight_layout()
        plt.savefig('../img/mcycle_experiments/initial_samples')

        for i in progress:

            _ = s.run(dtrain_step)
            _, gloss_, z_, ref_samples_, mask_ = s.run(
                [gtrain_step, gloss, z, ref_samples, mask])

            prc_used = np.sum(mask_) / len(mask_)

            progress.set_description('gloss=%.3f, prc_used=%.3f' % (
                gloss_, prc_used))

            all_gloss.append(gloss_)
            smoothed_gloss = np.mean(all_gloss[-10:])

            if smoothed_gloss < best_gloss:
                best_gloss = smoothed_gloss
                num_iter_no_improve = 0

            else:
                num_iter_no_improve += 1

            if tol_num_iter_no_improve is not None:
                if num_iter_no_improve > tol_num_iter_no_improve:
                    progress.close()
                    break

        z_, ref_samples_, predictions_ = s.run([z, ref_samples, predictions])

    fig, ax = plt.subplots(ncols=1, nrows=4, sharex=True, figsize=(6, 8))

    ax[0].scatter(tx, ty)
    ax[0].set_ylabel('data (y)')
    ax[0].set_xlabel('data (x)')
    ax[1].hist(ref_samples_, bins=20)
    ax[1].set_ylabel('number of samples')
    ax[1].set_xlabel('reference samples')
    ax[2].hist(z_, bins=20)
    ax[2].set_ylabel('number of samples')
    ax[2].set_xlabel('final z values')
    ax[3].scatter(tx, ty)
    ax[3].plot(x_grid, np.mean(predictions_, axis=0))
    ax[3].set_ylabel('predict')

    plt.tight_layout()
    plt.savefig('../img/mcycle_experiments/final_samples.png')


def sample_from_initial_mm(tx, ty, savefile=None):

    mdl = MixtureModel(
        tx, 4, 1000,
        tau_initial=1e-4)  # n_clusters, batch_size

    z = mdl.samples

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        z_ = s.run(z)

    if savefile is not None:

        fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True)

        ax[0].scatter(tx, ty)
        ax[0].set_ylabel('data (y)')
        ax[0].set_xlabel('data (x)')
        ax[1].hist(z_, bins=20)
        ax[1].set_ylabel('number of samples')
        ax[1].set_xlabel('z values')

        plt.savefig(savefile)


def get_khps(tx, ty, batch_size=20, train_len=100):

    noise = tf.Variable(.1 * np.random.rand(batch_size) + .1, dtype=tf.float32)
    sfs = tf.Variable(1. * np.random.rand(batch_size) + .1, dtype=tf.float32)
    sls = tf.Variable(3. * np.ones((batch_size, 1)) + 1., dtype=tf.float32)

    mdl = SGPModel(tx, ty, jitter_magnitude=1e-5)
    mdl.set_khps(sls, sfs, noise)

    fullgp_nlogm = mdl.fullgp_nlog_marglik()

    train_step = tf.train.AdamOptimizer().minimize(fullgp_nlogm)

    with tf.Session() as s:

        s.run(tf.global_variables_initializer())

        for i in range(train_len):
            s.run(train_step)

        noise_, sfs_, sls_ = s.run([noise, sfs, sls])

    return np.median(noise_), np.median(sfs_), np.median(np.squeeze(sls_))


def check_approx_marglik(tx, ty, noise_, sfs_, sls_, z_, savefile=None):

    z = tf.constant(z_[:, np.newaxis, np.newaxis], dtype=tf.float32)

    noise = tf.constant(noise_ * np.ones(len(z_)), dtype=tf.float32)
    sfs = tf.constant(sfs_ * np.ones(len(z_)), dtype=tf.float32)
    sls = tf.constant(sls_ * np.ones((len(z_), 1)), dtype=tf.float32)

    mdl = SGPModel(tx, ty, jitter_magnitude=1e-5)
    mdl.set_khps(sls, sfs, noise)
    mdl.set_inducing_inputs(z)

    # mdl.set_kernel(sls, sfs)
    # mdl.set_noise(noise)

    # mdl.set_inducing_inputs(z)

    nlogm = mdl.nlog_marglik()

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        nlogm_ = s.run(nlogm)

    if savefile is not None:

        fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True)

        ax[0].scatter(tx, ty)
        ax[0].set_ylabel('data (y)')
        ax[0].set_xlabel('data (x)')
        ax[1].scatter(z_, np.squeeze(nlogm_))
        ax[1].set_ylabel('nlog_marglik')
        ax[1].set_xlabel('inducing input placement')

        plt.savefig(savefile)


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


if __name__ == '__main__':
    main()
