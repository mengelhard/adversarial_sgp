import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


class MixtureModel:

    def __init__(self, x, n_clusters, batch_size=None,
                 initializer='equal_kmeans',
                 mixture_weights_gen_type='gumbel-softmax',
                 tau_initial=None, learn_tau=False,
                 weights_nn_depth=None, weights_nn_width=None,
                 fix_mixture_components=False, fix_mixture_weights=False):

        assert x.ndim == 2
        self.num_x, self.dim_x = np.shape(x)
        self.x = tf.constant(x, dtype=tf.float32)  # [tf.newaxis, :, :]

        assert batch_size is not None
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

    def _build_model(self):

        with tf.variable_scope('mixture_weights'):
            self._build_weights_generator()

        with tf.variable_scope('mixture_components'):
            self._build_component_distributions()

        eps_gauss = tf.random_normal(
            (self.batch_size, self.initial_model['n_dims']),
            dtype=tf.float32)

        self.z = tf.reduce_sum(
            self.mixture_weights[:, :, tf.newaxis] * self.means[tf.newaxis, :, :],
            axis=1)
        self.z += eps_gauss * tf.reduce_sum(
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
        widths.append(np.sqrt(np.sum((pts - center)**2, axis=0)))

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


def sample_from_initial_model():

    from load_data import load_dataset
    import matplotlib.pyplot as plt

    tx, ty, vx, vy, mx, my = load_dataset('mcycle', val_prc=.1)
    data = np.concatenate([tx, ty[:, np.newaxis]], axis=1)

    batch_size = 200
    n_components = 20

    mm = MixtureModel(
        data, n_components, batch_size=batch_size,
        tau_initial=1e-6, learn_tau=False,
        initializer='equal_kmeans')

    z = mm.z

    with tf.Session() as s:

        s.run(tf.global_variables_initializer())
        z_ = s.run(z)

    fig, axs = plt.subplots(ncols=2, nrows=1,
                            sharey=True, figsize=(12, 6))

    axs[0].scatter(data[:, 0], data[:, 1])
    axs[0].set_title('Training Data')
    axs[1].scatter(z_[:, 0], z_[:, 1])
    axs[1].set_title('%i Samples from GMM with %i Components' % (
        batch_size, n_components))

    plt.savefig('../img/mm_test/sample_from_initial_model.png')


def test_mixture_model():

    sample_from_initial_model()


if __name__ == '__main__':
    test_mixture_model()
