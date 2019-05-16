import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


class LearnWithUnnormalized:

    def __init__(self, u, q_samples,
                 ref_samples=None, ref_logprob=None,
                 ref_data=None, ref_num_gaussians=None):

        self.u = u
        self.q = q

        if (ref_samples is not None) and ref_logprob is not None:

            self.ref_samples = ref_samples
            self.ref_logprob = ref_logprob

        elif (ref_data is not None) and ref_num_gaussians is not None:

            self._create_gaussian_reference_from_data(
                ref_data, ref_num_gaussians)

        self._build_adversary()
        self._get_train_step()

    def _build_adversary(self):



    def _get_train_step(self):

        dloss_total = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=dy, labels=tf.ones_like(dy))
        dloss_total += tf.nn.sigmoid_cross_entropy_with_logits(
            logits=dref, labels=tf.zeros_like(dref))
        dloss = tf.reduce_mean(dloss_total)

        gloss = tf.reduce_sum(ref_logprob + target_distribution + dz, axis=0)

        dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "adversary")
        gvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")

        dtrain_step = tf.train.AdamOptimizer(FLAGS.d_lr).minimize(dloss, var_list=dvars)
        gtrain_step = tf.train.AdamOptimizer(FLAGS.g_lr).minimize(gloss, var_list=gvars)

    return sls, sfs, noise, weights, mse, nmse, dloss, gloss, z

    def _create_gaussian_reference_from_data(self, data, num_gaussians):

        self.ref_logprob, self.ref_samples = scoping_gaussians_reference(
            data, num_gaussians)




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


def scoping_gaussians_reference(d, num_gaussians):

    mmref = kmeans_mixture_model(d, n_clusters=1)

    ref = distributions.MixtureSameFamily(
        mixture_distribution=distributions.Categorical(
            probs=np.ones(num_gaussians) / num_gaussians),
        components_distribution=distributions.MultivariateNormalDiag(
            loc=np.array([mmref['means'][0]] * num_gaussians).astype(np.float32),
            scale_diag=np.array(
                [(scale + 1) * mmref['widths'][0]
                 for scale in range(num_gaussians)]).astype(np.float32)
        )
    )

    return ref.log_prob, ref.sample


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
