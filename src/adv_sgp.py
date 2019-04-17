import numpy as np
import tensorflow as tf
import os
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.contrib import slim
from tensorflow.contrib import distributions
from tqdm import tqdm
from load_data import load_dataset
from sgp import nlog_sgp_marglik, sgp_pred, scaled_square_dist
from vis_data import training_summary, vis_gen
matplotlib.use('Agg')

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('img_path', '../img', """Directory for output figs""")
tf.app.flags.DEFINE_integer('niter', 10000, """Number of iterations""")
tf.app.flags.DEFINE_integer('burn_in', 100, """Head-start for discriminator""")
tf.app.flags.DEFINE_integer('print_freq', 500, """How often to display results""")
tf.app.flags.DEFINE_float('g_lr', 1e-4, """Generator learning rate""")
tf.app.flags.DEFINE_float('d_lr', 1e-4, """Discriminator learning rate""")
tf.app.flags.DEFINE_integer('batch_size', 200, """Batch size""")
tf.app.flags.DEFINE_string('gen_type', 'simple', """Generator Type: simple, direct, or gumbel""")
tf.app.flags.DEFINE_string('dataset', 'kin40k', """Dataset to Use""")
tf.app.flags.DEFINE_float('val_prc', .2, """Portion of training data to use for validation""")
tf.app.flags.DEFINE_integer('n_clusters', 50, """Number of mixture model clusters""")
tf.app.flags.DEFINE_integer('n_z', 100, """Number of pseudo-inputs""")
tf.app.flags.DEFINE_string('sgp_approx', 'vfe', """SGP approximation method""")
tf.app.flags.DEFINE_float('sls', 1., """Initial GP length scale""")
tf.app.flags.DEFINE_float('sfs', 1., """Initial GP function scale""")
tf.app.flags.DEFINE_float('noise', 1e-3, """Initial GP noise""")
tf.app.flags.DEFINE_float('lbd', 0., """Hyperparameter penalty""")
tf.app.flags.DEFINE_integer('seed', 1, """Seed (for clustering)""")
tf.app.flags.DEFINE_bool('ard', True, """Whether to use ARD kernel""")
tf.app.flags.DEFINE_integer('num_refs', 3, """How many reference Gaussians""")
tf.app.flags.DEFINE_integer('num_sgp_samples', 20, """How many SGP samples""")
tf.app.flags.DEFINE_float('min_dist', 3e-3, """Minimum scaled separation between pseudo-inputs""")


def mixture_pdf(model):
    dist = distributions.MixtureSameFamily(
        mixture_distribution=distributions.Categorical(
            probs=model['weights']),
        components_distribution=distributions.MultivariateNormalDiag(
            loc=model['means'],       # One for each component.
            scale_diag=model['sds']))  # And same here.
    return dist


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, num_samples=1):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    # logits = tf.tile(logits, (num_samples, 1))
    y = logits[tf.newaxis, :] + sample_gumbel(
        (num_samples, tf.shape(logits)[0]))
    return tf.nn.softmax(y / temperature, axis=1)  # check dimension


def mix_weights(n_components, gen_type='simple', tau_initial=1e-4,
                learn_tau=False, depth=3, width=None):

    if width is None:
        width = n_components

    with tf.variable_scope("mix_weights") as scope:

        if learn_tau:
            tau = slim.model_variable(
                'tau', shape=np.shape(tau_initial),
                initializer=tf.constant_initializer(tau_initial))
        else:
            tau = tau_initial

        if gen_type == 'simple':
            w = slim.model_variable(
                'weightlogits', shape=(n_components),
                initializer=tf.zeros_initializer())
            weights = gumbel_softmax_sample(
                w, temperature=tau,
                num_samples=FLAGS.batch_size)

        elif gen_type == 'direct':
            net = tf.random_normal(
                (FLAGS.batch_size, width),
                dtype=tf.float32)
            for i in range(depth - 1):
                net = slim.fully_connected(
                    net, width,
                    activation_fn=tf.nn.elu, scope='fc_%d' % (i + 1))
            net = slim.fully_connected(
                net, n_components,
                activation_fn=tf.nn.elu, scope='fc_final')
            weights = slim.softmax(net / tau)

        elif gen_type == 'gumbel':
            net = tf.random_normal(
                (width),
                dtype=tf.float32)
            for i in range(depth - 1):
                net = slim.fully_connected(
                    net, width,
                    activation_fn=tf.nn.elu, scope='fc_%d' % (i + 1))
            net = slim.fully_connected(
                net, n_components,
                activation_fn=tf.nn.elu, scope='fc_final')
            w = tf.nn.log_softmax(net, axis=1)
            weights = gumbel_softmax_sample(
                w, temperature=tau,
                num_samples=FLAGS.batch_size)

    return weights


def generator(initial_model, type='simple', tau_initial=1e-4,
              learn_tau=True, nn_depth=3, reuse=False):
    """
    Use random noise 'eps' to sample from mixture model
    """

    with tf.variable_scope("generator", reuse=reuse) as scope:

        means = slim.model_variable(
            'means', shape=np.shape(initial_model['means']),
            initializer=tf.constant_initializer(initial_model['means']))

        sds = slim.model_variable(
            'sds', shape=np.shape(initial_model['sds']),
            initializer=tf.constant_initializer(initial_model['sds']))

        weights = mix_weights(
            initial_model['n_components'], gen_type='simple', tau_initial=1e-4,
            learn_tau=False, depth=3, width=None)

        eps_gauss = tf.random_normal(
            (FLAGS.batch_size, initial_model['n_dims']),
            dtype=tf.float32)

        y = tf.reduce_sum(
            weights[:, :, tf.newaxis] * means[tf.newaxis, :, :],
            axis=1)
        y += tf.reduce_sum(
            weights[:, :, tf.newaxis] * sds[tf.newaxis, :, :],
            axis=1) * eps_gauss

        return y, weights


def gen_lognormal(var_name, num_samples, initial_mean, initial_sd=1.):

    with tf.variable_scope(var_name):

        mean = slim.model_variable(
            'mean', shape=(),
            initializer=tf.constant_initializer(np.log(initial_mean)))
        sd = slim.model_variable(
            'sd', shape=(),
            initializer=tf.constant_initializer(initial_sd))
        eps = tf.random_normal(shape=(num_samples), dtype=tf.float32)

    return tf.exp(eps * sd[tf.newaxis] + mean[tf.newaxis])


def khp_generator(num_samples, num_dims, ard=FLAGS.ard,
                  initial_sls=FLAGS.sls, initial_sfs=FLAGS.sfs,
                  initial_noise=FLAGS.noise):

    with tf.variable_scope('khp_generator', reuse=reuse) as scope:

        if ard:
            sls = tf.stack([gen_lognormal('sls_%i' % x, num_samples, initial_sls)
                            for x in range(num_dims)], axis=1)
        else:
            sls = gen_lognormal('sls', num_samples, initial_sls)[:, tf.newaxis]

        sfs = gen_lognormal('sfs', num_samples, initial_sfs)
        noise = gen_lognormal('noise', num_samples, initial_noise)

    return sls, sfs, noise


def adversary(y, reuse=False):

    with tf.variable_scope('adversary', reuse=reuse):

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


def kmeans_mixture_model(data, n_clusters=100, random_state=0):
    """
    Cluster x and return cluster centers and cluster widths (variances)
    """
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state).fit(data)

    lbls = km.labels_
    cc = km.cluster_centers_

    x_centered = np.array([x - cc[y] for x, y in zip(data, lbls)])
    c_widths = np.array(
        [np.sum(x_centered[lbls == c, :]**2, axis=0) / np.sum(lbls == c)
         for c in range(n_clusters)])

    weights = np.array([np.sum(lbls == c) for c in range(n_clusters)])
    weights = weights / np.sum(weights)

    mixture_model = {
        'n_components': n_clusters,
        'n_dims': np.shape(data)[1],
        'weights': weights,
        'means': cc,
        'sds': np.sqrt(c_widths)}

    return mixture_model


def reference(d, num_refs=FLAGS.num_refs):

    mmref = kmeans_mixture_model(d, n_clusters=1)

    ref = distributions.MixtureSameFamily(
        mixture_distribution=distributions.Categorical(
            probs=[.2] * num_refs),
        components_distribution=distributions.MultivariateNormalDiag(
            loc=np.array([mmref['means'][0]] * num_refs).astype(np.float32),
            scale_diag=np.array(
                [mmref['sds'][0] * scale
                 for scale in [2**e for e in range(num_refs)]]).astype(np.float32)
        )
    )

    return ref.log_prob, ref.sample


def random_samples(x, rows_per_sample, n_samples, axis=0):
    indices = tf.random_uniform((n_samples, tf.shape(x)[0]), dtype=tf.float32)
    _, indices = tf.nn.top_k(indices, k=rows_per_sample)
    return tf.gather(x, indices), indices


def get_mse(pred, real):  # check axes
    se = (pred - real)**2
    ms = tf.reduce_mean(se)
    _, var = tf.nn.moments(real, axes=[0])
    return ms, ms / var


def trainplot(val_freq, val_error, train_freq=None, train_error=None, savepath=None):

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    val_iters = val_freq * np.arange(len(val_error))
    ax.plot(val_iters, val_error)
    if train_error is not None:
        train_iters = train_freq * np.arange(len(train_error))
        ax.plot(train_iters, train_error)
    ax.set_ylabel('SMSE')
    ax.set_xlabel('Iteration')

    if savepath is not None:
        plt.savefig(os.path.join(savepath, 'training.png'))
    else:
        plt.show()


def sgp_model(tx, ty, vx, vy):

    initial_model = kmeans_mixture_model(tx, n_clusters=FLAGS.n_clusters)

    y, weights = generator(initial_model, type=FLAGS.gen_type)

    sls, sfs, noise = khp_generator(FLAGS.num_sgp_samples, np.shape(tx)[1])

    # reference distribution: need something really wide -- scoping gaussians?
    reflogprob, rs = reference(tx, num_refs=FLAGS.num_refs)
    refsample = tf.cast(rs(sample_shape=FLAGS.batch_size), dtype=tf.float32)

    # specify the adversary, which will learn the likelihood ratio
    # between generator samples and reference samples

    dref = adversary(refsample)
    dy = adversary(y, reuse=True)

    # Get pseudo-inputs
    z, z_idx = random_samples(y, FLAGS.n_z, FLAGS.num_sgp_samples)  # samples of (full sets of) pseudo-inputs

    # Make sure pseudo-inputs aren't too close together
    # NEED TO DOUBLE CHECK THIS?

    z, z_idx, sls, sfs, noise = z_mask(
        z, z_idx, sls, sfs, noise)

    # get target distribution (SGP marginal likelihood)
    target_distribution = nlog_sgp_marglik(
        tx, ty, z, sls, sfs, noise,
        obj=FLAGS.sgp_approx)

    # get likelihood ratio of reference samples
    # IMPORTANT: DO WE NEED TO INCLUDE KHP DETAILS HERE TOO? ###
    ref_logprob = tf.reduce_sum(reflogprob(z), axis=1)
    dz = tf.reduce_sum(tf.gather(dy, z_idx), axis=1)

    # predictions for test set / val set
    pred = sgp_pred(tx, ty, vx, z, sls, sfs, noise, obj=FLAGS.sgp_approx)

    # Get MSE / normalized MSE
    mse, nmse = get_mse(tf.squeeze(pred), tf.constant(vy, dtype=tf.float32))

    dloss_total = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=dy, labels=tf.ones_like(dy))
    dloss_total += tf.nn.sigmoid_cross_entropy_with_logits(
        logits=dref, labels=tf.zeros_like(dref))
    dloss = tf.reduce_mean(dloss_total)

    gloss = tf.reduce_sum(ref_logprob + target_distribution + dz, axis=0)

    return sls, sfs, noise, weights, mse, nmse, dloss, gloss, z


def z_mask(z, z_idx, sls, sfs, noise):

    ss_dist = scaled_square_dist(z, z, sls)
    ss_dist = ss_dist + tf.eye(FLAGS.n_z, dtype=tf.float32)[tf.newaxis, :, :]
    mask = tf.reduce_min(ss_dist, axis=[1, 2]) > FLAGS.min_dist
    z = tf.boolean_mask(z, mask)
    z_idx = tf.boolean_mask(z_idx, mask)
    sls = tf.boolean_mask(sls, mask)
    sfs = tf.boolean_mask(sfs, mask)
    noise = tf.boolean_mask(noise, mask)

    return z, z_idx, sls, sfs, noise


def main(argv):

    # Load data

    train_x, train_y, val_x, val_y, test_x, test_y = load_dataset(
        FLAGS.dataset, val_prc=FLAGS.val_prc)

    # CHANGE THIS LINE WHEN USING A VALIDATION SET

    sls, sfs, noise, weights, mse, nmse, dloss, gloss, z = sgp_model(
        train_x, train_y, test_x, test_y)

    # prepare for run

    dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "adversary")
    gvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")

    dtrain_step = tf.train.AdamOptimizer(FLAGS.d_lr).minimize(dloss, var_list=dvars)
    gtrain_step = tf.train.AdamOptimizer(FLAGS.g_lr).minimize(gloss, var_list=gvars)

    outdir = FLAGS.img_path
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    def run_training(sess, niter=FLAGS.niter, ninitial=FLAGS.burn_in,
                     figdir=FLAGS.img_path, cluster=False):

        # consider a burn-in period for the discriminator before going to the generator

        if not cluster:
            burn_in = tqdm(range(ninitial))

        for i in range(ninitial):  # burn_in:

            dloss_, _ = sess.run([dloss, dtrain_step])

            if not cluster:
                burn_in.set_description("dloss=%.3f" % (dloss_))

        printfreq = FLAGS.print_freq

        if not cluster:
            progress = tqdm(range(niter))

        summary = training_summary(['mse', 'nmse', 'dloss', 'gloss'])

        for i in range(niter):  # progress:

            dloss_, _ = sess.run([dloss, dtrain_step])
            gloss_, _ = sess.run([gloss, gtrain_step])

            summary.add_point('dloss', [i, dloss_])
            summary.add_point('gloss', [i, gloss_])

            if i % printfreq == 0:

                nmse_, mse_, z_ = sess.run([nmse, mse, z])
                summary.add_point('nmse', [i, nmse_])
                summary.add_point('mse', [i, mse_])

                fig, ax = plt.subplots(nrows=1, ncols=1)
                vis_gen(FLAGS.dataset, ax, z_, train_x, train_y)
                plt.savefig(os.path.join(FLAGS.img_path, 'z_%i.png' % i))

            if not cluster:
                progress.set_description("dloss=%.3f,gloss=%.3f,nmse=%.3f" % (
                    dloss_, gloss_, nmse_))

        fig, ax = plt.subplots(nrows=2, ncols=1)
        summary.plot_metrics(ax[0], ['nmse', 'mse'])
        summary.plot_metrics(ax[1], ['gloss'])
        plt.savefig(os.path.join(FLAGS.img_path, 'training.png'))

        # trainplot(printfreq, nms_error, savepath=FLAGS.img_path)

    try:
        s.close()
    except NameError:
        pass
    s = tf.InteractiveSession()
    s.run(tf.global_variables_initializer())
    run_training(s, niter=FLAGS.niter)


if __name__ == '__main__':
    tf.app.run()
