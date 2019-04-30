
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sgp import SGPModel
from tqdm import tqdm
from load_data import load_dataset


def main():

    tx, ty, vx, vy, mx, my = load_dataset('mcycle', val_prc=.2)

    initial_z = np.random.rand(16, 3, 1) * 60.

    with tf.variable_scope('sgp_model'):

        z = tf.Variable(initial_z, name='z', dtype=tf.float32)

        # z = tf.constant(initial_z, name='z', dtype=tf.float32)

        sls = tf.exp(tf.Variable(
            2. * np.ones((16, 1)), name='sls', dtype=tf.float32))
        sfs = tf.exp(tf.Variable(
            1. * np.ones(16), name='sfs', dtype=tf.float32))
        noise = tf.exp(tf.Variable(
            -1. * np.ones(16), name='noise', dtype=tf.float32))

        # sls = tf.constant(np.ones((16, 1)), name='sls', dtype=tf.float32)
        # sfs = tf.constant(np.ones(16), name='sfs', dtype=tf.float32)
        # noise = tf.constant(.1 * np.ones(16), name='noise', dtype=tf.float32)

        sgpm = SGPModel(tx, ty, jitter_magnitude=3e-6, approx_method='fitc')
        sgpm.set_kernel(sls, sfs)
        sgpm.set_noise(noise)
        sgpm.set_inducing_inputs(z)

    loss = sgpm.nlog_marglik()
    train_step = tf.train.AdamOptimizer(3e-4).minimize(loss)

    predictions = sgpm.predict(mx)

    plotpoints = np.linspace(0, 60, 200)[:, np.newaxis]

    fgp_plotpoints = sgpm.fullgp_predict(plotpoints)
    sgp_plotpoints = sgpm.predict(plotpoints)

    mse, nmse = get_mse(predictions, tf.constant(my, dtype=tf.float32))

    num_iterations = 10000
    print_freq = 20

    with tf.Session() as s:

        s.run(tf.global_variables_initializer())

        progress = tqdm(range(num_iterations))

        for i in progress:

            loss_, _ = s.run([loss, train_step])
            progress.set_description('nll = %.3f' % np.mean(loss_))

            if (i % print_freq) == (print_freq - 1):

                mse_ = s.run(mse)
                print('\nMSE is %.5f' % mse_)

        predictions_, sgp_plotpoints_, fgp_plotpoints_, z_ = s.run(
            [predictions, sgp_plotpoints, fgp_plotpoints, z])

    fig, axs = plt.subplots(ncols=4, nrows=4, figsize=(20, 20))

    for (i, j), ax in np.ndenumerate(axs):

        ax.plot(np.squeeze(plotpoints), np.squeeze(sgp_plotpoints_[i * 4 + j, :]))
        ax.plot(np.squeeze(plotpoints), np.squeeze(fgp_plotpoints_[i * 4 + j, :]))
        ax.scatter(np.squeeze(tx), ty)
        ax.scatter(z_[i * 4 + j, :], -3. * np.ones(len(z_[i * 4 + j, :])))
        ax.legend(['sgp', 'fgp', 'data', 'z'])

    plt.savefig('../img/z_plots.png')


def get_mse(pred, real):  # check axes
    se = (pred - real)**2
    ms = tf.reduce_mean(se)
    _, var = tf.nn.moments(real, axes=[0])
    return ms, ms / var


if __name__ == '__main__':

    main()
