import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib import distributions
from load_data import load_dataset
from sgp import SGPModel
from tqdm import tqdm


class MixtureModel:

    def __init__(self, x, n_clusters, batch_size,
                 initializer='kmeans',
                 mixture_weights_gen_type='gumbel-softmax',
                 tau_initial=None, learn_tau=False,
                 weights_nn_depth=None, weights_nn_width=None,
                 fix_mixture_components=False, fix_mixture_weights=False):

        assert x.ndim == 2
        self.num_x, self.dim_x = np.shape(x)
        self.x = tf.constant(x, dtype=tf.float32)  # [tf.newaxis, :, :]

        self.n_clusters = n_clusters
        self.batch_size = batch_size

        self._mixture_weights_gen_type = mixture_weights_gen_type
        self._tau_initial = tau_initial
        self._learn_tau = learn_tau
        self._weights_nn_depth = weights_nn_depth
        self._weights_nn_width = weights_nn_width

        self._fix_mixture_components = fix_mixture_components
        self._fix_mixture_weights = fix_mixture_weights

        if initializer == 'kmeans':
            self.initial_model = kmeans_mixture_model(
                x, n_clusters=n_clusters)
        elif initializer == 'equal_kmeans':
            self.initial_model = equal_kmeans_mixture_model(
                x, n_clusters=n_clusters)
        elif initializer == 'subset_of_data':
            self.initial_model = subset_of_data_mixture_model(
                x, n_clusters=n_clusters)

        with tf.variable_scope('mixture_model'):
            self._build_model()

        with tf.variable_scope('reference'):
            self._build_reference()

    def _build_model(self):

        with tf.variable_scope('mixture_weights'):
            self._build_weights_generator()

        with tf.variable_scope('mixture_components'):
            self._build_component_distributions()

        eps_gauss = tf.random_normal(
            (self.batch_size, self.initial_model['n_dims']),
            dtype=tf.float32)

        self.samples = tf.reduce_sum(
            self.mixture_weights[:, :, tf.newaxis] * self.means[tf.newaxis, :, :],
            axis=1)
        self.samples += eps_gauss * tf.reduce_sum(
            self.mixture_weights[:, :, tf.newaxis] * self.widths[tf.newaxis, :, :],
            axis=1)

    def _build_weights_generator(self):

        if self._fix_mixture_weights:
            self.mixture_weights = tf.constant(
                self.initial_model['weights'],
                dtype=tf.float32)

        elif self._mixture_weights_gen_type == 'gumbel-softmax':
            self._build_weights_gen_gs()

        elif self._mixture_weights_gen_type == 'nn-gumbel-softmax':
            self._build_weights_gen_nngs()

        elif self._mixture_weights_gen_type == 'nn':
            self._build_weights_gen_nn()

    def _build_component_distributions(self):

        if self._fix_mixture_components:

            self.means = tf.constant(
                self.initial_model['means'],
                name='means',
                dtype=tf.float32)

            self.widths = tf.constant(
                self.initial_model['widths'],
                name='widths',
                dtype=tf.float32)

        else:

            self.means = tf.Variable(
                self.initial_model['means'],
                name='means',
                dtype=tf.float32)

            self.widths = tf.Variable(
                self.initial_model['widths'],
                name='widths',
                dtype=tf.float32)

    def _build_weights_gen_gs(self):

        assert self._tau_initial is not None
        assert self._learn_tau is not None

        if self._learn_tau:
            self.tau = tf.Variable(
                self._tau_initial, name='tau', dtype=tf.float32)
        else:
            self.tau = tf.constant(
                self._tau_initial, name='tau', dtype=tf.float32)

        weight_logits = tf.Variable(
            np.zeros(self.initial_model['n_components']),
            name='weightlogits',
            dtype=tf.float32)

        self.mixture_weights = gumbel_softmax_sample(
            weight_logits,
            temperature=self.tau,
            num_samples=self.batch_size)

    def _build_weights_gen_nngs(self):

        assert self._weights_nn_depth > 0

        if self._weights_nn_width is None:
            self._weights_nn_width = self.initial_model['n_components']

        net = tf.random_normal(
            (self._weights_nn_width), dtype=tf.float32)

        for i in range(self._weights_nn_depth - 1):

            net = layers.fully_connected(
                net, self._weights_nn_width,
                activation_fn=tf.nn.elu, scope='fc_%d' % (i + 1))

        net = layers.fully_connected(
            net, self.initial_model['n_components'],
            activation_fn=tf.nn.elu, scope='fc_final')

        weight_logits = tf.nn.log_softmax(net, axis=1)

        self.mixture_weights = gumbel_softmax_sample(
            weight_logits,
            temperature=self.tau,
            num_samples=self.batch_size)

    def _build_weights_gen_nn(self):  # STOPPED HERE

        assert self._weights_nn_depth > 0

        if self._weights_nn_width is None:
            self._weights_nn_width = self.initial_model['n_components']

        net = tf.random_normal(
            (self.batch_size, self._weights_nn_width),
            dtype=tf.float32)

        for i in range(self._weights_nn_depth - 1):

            net = layers.fully_connected(
                net, self._weights_nn_width,
                activation_fn=tf.nn.elu, scope='fc_%d' % (i + 1))

        net = layers.fully_connected(
            net, self.initial_model['n_components'],
            activation_fn=tf.nn.elu, scope='fc_final')

        self.mixture_weights = tf.nn.softmax(net / self.tau, axis=1)

    def _build_reference(self):

        reference_components = [distributions.MultivariateNormalDiag(
            loc=self.means[i, :],
            scale_diag=self.widths[i, :]) for i in range(self.n_clusters)]

        reference_probs = tf.reduce_mean(self.mixture_weights, axis=0)

        ref = distributions.Mixture(
            cat=distributions.Categorical(probs=reference_probs),
            components=reference_components
        )

        self.ref_logprob_gen_samples = ref.log_prob(self.samples)
        self.ref_samples = ref.sample(sample_shape=self.batch_size)


class SGPGenerator(MixtureModel):

    def __init__(self, x, n_clusters,
                 num_inducing_inputs,
                 num_sgp_samples,
                 initializer='equal_kmeans',
                 mixture_weights_gen_type='gumbel-softmax',
                 tau_initial=None, learn_tau=False,
                 weights_nn_depth=None, weights_nn_width=None,
                 fix_mixture_components=False, fix_mixture_weights=False,
                 initial_sls=0., initial_sfs=0., initial_noise=-5.):

        self._num_z = num_inducing_inputs
        self._num_sgp_samples = num_sgp_samples
        mixture_model_batch_size = self._num_z * self._num_sgp_samples

        self._initial_sls = initial_sls
        self._initial_sfs = initial_sfs
        self._initial_noise = initial_noise

        MixtureModel.__init__(self, x, n_clusters, mixture_model_batch_size,
                              initializer=initializer,
                              mixture_weights_gen_type=mixture_weights_gen_type,
                              tau_initial=tau_initial, learn_tau=learn_tau,
                              weights_nn_depth=weights_nn_depth,
                              weights_nn_width=weights_nn_width,
                              fix_mixture_components=fix_mixture_components,
                              fix_mixture_weights=fix_mixture_weights)

        self._build_khp_gen()
        self.z = tf.reshape(
            self.samples,
            [self._num_sgp_samples, self._num_z, self.dim_x])

    def _build_khp_gen(self):

        assert np.ndim(self._initial_sls) < 2
        assert np.ndim(self._initial_sfs) == 0
        assert np.ndim(self._initial_noise) == 0

        with tf.variable_scope('khp'):

            if np.ndim(self._initial_sls) == 1:

                assert len(self._initial_sls) == self.dim_x

                all_sls = [gen_lognormal(
                    self._num_sgp_samples,
                    ls,
                    scope='sls_%i' % i) for i, ls in enumerate(self._initial_sls)]

                self.sls = tf.stack(all_sls, axis=1)

            else:

                self.sls = gen_lognormal(
                    self._num_sgp_samples,
                    self._initial_sls,
                    scope='sls')[:, tf.newaxis]

            self.sfs = gen_lognormal(
                self._num_sgp_samples,
                self._initial_sfs,
                scope='sfs')

            self.noise = gen_lognormal(
                self._num_sgp_samples,
                self._initial_noise,
                scope='noise')


def kmeans_mixture_model(d, n_clusters, random_state=0):
    """
    Cluster x and return cluster centers and cluster widths
    """
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state).fit(d)

    lbl = km.labels_
    unique_lbl = np.unique(lbl)

    weights = np.array([np.sum(lbl == i) for i in unique_lbl]) / len(lbl)
    centers = np.array([np.mean(d[lbl == i, :], axis=0) for i in unique_lbl])
    widths = get_cluster_widths(d, lbl, centers)

    mixture_model = {
        'n_components': n_clusters,
        'n_dims': np.shape(d)[1],
        'weights': weights,
        'means': centers,
        'widths': widths}

    return mixture_model


def equal_kmeans_mixture_model(d, n_clusters, random_state=0):

    from ekm import ekm_unequal_clusters

    d, lbl = ekm_unequal_clusters(d, n_clusters)
    unique_lbl = np.unique(lbl)

    weights = np.array([np.sum(lbl == i) for i in unique_lbl]) / len(lbl)
    centers = np.array([np.mean(d[lbl == i, :], axis=0) for i in unique_lbl])
    widths = get_cluster_widths(d, lbl, centers)

    mixture_model = {
        'n_components': n_clusters,
        'n_dims': np.shape(d)[1],
        'weights': weights,
        'means': centers,
        'widths': widths}

    return mixture_model


def subset_of_data_mixture_model(d, n_clusters, random_state=0):

    weights = np.ones(n_clusters, dtype=np.float32) / n_clusters
    centers = np.random.permutation(d)[:n_clusters, :]
    lbl = np.argmin(pw_dist(d, centers), axis=1)
    widths = get_cluster_widths(d, lbl, centers)

    mixture_model = {
        'n_components': n_clusters,
        'n_dims': np.shape(d)[1],
        'weights': weights,
        'means': centers,
        'widths': widths}

    return mixture_model


def get_cluster_widths(points, labels, centers):

    widths = []

    for label, center in zip(np.unique(labels), centers):

        pts = points[labels == label, :]
        widths.append(np.sqrt(np.mean((pts - center)**2, axis=0)))

    return np.array(widths)


def pw_dist(x, y):
    diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))


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


def gen_lognormal(num_samples, initial_mean, initial_sd=1., scope=None):

    with tf.variable_scope(scope):

        mean = tf.Variable(
            initial_mean,
            name='mean',
            dtype=tf.float32)

        sd = tf.Variable(
            initial_sd,
            name='sd',
            dtype=tf.float32)

        eps = tf.random_normal(shape=[num_samples], dtype=tf.float32)

    return tf.exp(eps * sd[tf.newaxis] + mean[tf.newaxis])


# Model Tests #


def sample_from_initial_sgp_generator(tx, ty, vx, vy):

    plt.scatter(np.squeeze(tx), ty)
    plt.savefig('../img/mm_test/mcycle1.png')

    num_inducing_inputs = 4
    num_sgp_samples = 9
    n_components = 4

    # equal k-means is doing something weird to the array tx -- np shuffle?

    sgpgen = SGPGenerator(
        tx, n_components, num_inducing_inputs,
        num_sgp_samples,
        tau_initial=1e-6, learn_tau=False,
        initializer='kmeans')

    plt.scatter(np.squeeze(tx), ty)
    plt.savefig('../img/mm_test/mcycle2.png')

    z = sgpgen.z

    sgpm = SGPModel(tx, ty, jitter_magnitude=3e-6, approx_method='vfe')
    sgpm.set_kernel(sgpgen.sls, sgpgen.sfs)
    sgpm.set_noise(sgpgen.noise)
    sgpm.set_inducing_inputs(z)

    plt.scatter(np.squeeze(tx), ty)
    plt.savefig('../img/mm_test/mcycle3.png')

    # loss = sgpm.nlog_marglik()
    # train_step = tf.train.AdamOptimizer(3e-4).minimize(loss)

    sgp_predictions = sgpm.predict(vx)
    fgp_predictions = sgpm.fullgp_predict(vx)

    plotpoints = np.linspace(0, 60, 200)[:, np.newaxis]
    fgp_plotpoints = sgpm.fullgp_predict(plotpoints)
    sgp_plotpoints = sgpm.predict(plotpoints)

    fgp_mse, _ = get_mse(
        fgp_predictions,
        tf.constant(vy, dtype=tf.float32)[tf.newaxis, :])

    sgp_mse, _ = get_mse(
        sgp_predictions,
        tf.constant(vy, dtype=tf.float32)[tf.newaxis, :])

    with tf.Session() as s:

        s.run(tf.global_variables_initializer())

        sgp_mse_, fgp_mse_, sgp_plotpoints_, fgp_plotpoints_, z_ = s.run(
            [sgp_mse, fgp_mse, sgp_plotpoints, fgp_plotpoints, z])

    plt.scatter(np.squeeze(tx), ty)
    plt.savefig('../img/mm_test/mcycle4.png')

    fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(15, 15))

    for (i, j), ax in np.ndenumerate(axs):

        idx = i * 3 + j
        current_z = np.squeeze(z_[idx, :, :])

        ax.plot(np.squeeze(plotpoints), np.squeeze(sgp_plotpoints_[idx, :]))
        ax.plot(np.squeeze(plotpoints), np.squeeze(fgp_plotpoints_[idx, :]))
        ax.scatter(np.squeeze(tx), ty)
        ax.scatter(current_z, -3. * np.ones(len(current_z)))
        ax.legend(['sgp', 'fgp', 'data', 'z'])
        ax.set_title('SGP MSE: %.3f, FGP MSE: %.3f' % (sgp_mse_[idx], fgp_mse_[idx]))

    plt.savefig('../img/mm_test/sample_from_sgp_generator.png')


def sample_from_initial_mixture_model(data):

    batch_size = 200
    n_components = 20

    mm = MixtureModel(
        data, n_components, batch_size=batch_size,
        tau_initial=1e-6, learn_tau=False,
        initializer='equal_kmeans')

    samples = mm.samples

    with tf.Session() as s:

        s.run(tf.global_variables_initializer())
        samples_ = s.run(samples)

    fig, axs = plt.subplots(ncols=2, nrows=1,
                            sharey=True, figsize=(12, 6))

    axs[0].scatter(data[:, 0], data[:, 1])
    axs[0].set_title('Training Data')
    axs[1].scatter(samples_[:, 0], samples_[:, 1])
    axs[1].set_title('%i Samples from GMM with %i Components' % (
        batch_size, n_components))

    plt.savefig('../img/mm_test/sample_from_initial_mixture_model.png')


def test_mixture_model():

    tx, ty, vx, vy, mx, my = load_dataset('mcycle', val_prc=.2)
    mixture_model_data = np.concatenate([tx, ty[:, np.newaxis]], axis=1)

    # sample_from_initial_mixture_model(mixture_model_data)
    sample_from_initial_sgp_generator(tx, ty, vx, vy)


def get_mse(pred, real):  # check axes
    se = (pred - real)**2
    ms = tf.reduce_mean(se, axis=1)
    _, var = tf.nn.moments(real, axes=[1])
    return ms, ms / var


if __name__ == '__main__':
    test_mixture_model()
