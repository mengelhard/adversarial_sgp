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

    tx, ty, vx, vy, mx, my = load_dataset('kin40k', val_prc=.01)

    for num_ind_inputs in [32, 64, 128, 256, 512, 1024]:

        for gaussian_reference in [True, False]:

            for mw_alpha in [1., 10., 100.]:

                learn_asgp(
                    tx, ty, mx, my,
                    num_ind_inputs=num_ind_inputs,
                    gref=gaussian_reference,
                    mwa=mw_alpha)


def learn_asgp(tx, ty, mx, my,
               num_ind_inputs=None,
               gaussian_reference=None,
               mw_alpha=None):

    num_components = num_ind_inputs
    num_sgp_samples = 100
    num_steps = 300

    with tf.variable_scope('generator'):

        sgp_gen = SGPGenerator(
            tx, num_components, num_ind_inputs, num_sgp_samples,
            component_weights_gen_type='nn-gumbel-softmax',
            weights_nn_depth=4,
            tau_initial=1e-4,
            gaussian_reference=gaussian_reference)

    sgpm = SGPModel(tx, ty, jitter_magnitude=1e-4)

    asgp = ASGP(
        sgp_gen, sgpm,
        mw_alpha=mw_alpha,
        n0=1.,
        filter_dist=1e-4,
        g_lr=1e-1)

    mask = sgpm.mask

    predictions = sgpm.predict(mx)
    avg_prediction = tf.reduce_mean(predictions, axis=0)
    nmse = normalized_mean_square_error(avg_prediction, my)

    with tf.Session() as s:

        s.run(tf.global_variables_initializer())

        initial_z_, ref_samples_ = s.run([sgp_gen.samples, sgp_gen.ref_samples])

        for i in range(100):

            _ = s.run(asgp.dtrain_step)

        progress = tqdm(range(num_steps))

        # plot_kin_initial(tx, ty, ref_samples_, initial_z_)

        for i in progress:

            _ = s.run(asgp.dtrain_step)
            _, gloss_, mask_ = s.run(
                [asgp.gtrain_step, asgp.gloss, sgpm.mask])

            prc_used = np.sum(mask_) / len(mask_)

            progress.set_description('gloss=%.3f, prc_used=%.3f' % (
                gloss_, prc_used))

        z_, ref_samples_, predictions_, nmse_ = s.run(
            [sgp_gen.samples, sgp_gen.ref_samples, predictions, nmse])

    print('Normalized MSE: ', nmse_)

    # plot_kin_final(tx, ty, ref_samples_, z_, x_grid, predictions_)


def plot_kin_initial(tx, ty, ref_samples, z):

    return None


def plot_kin_final(tx, ty, ref_samples, z):

    return None


def normalized_mean_square_error(pred, real, return_mse=True):
    mse = mean_square_error(pred, real)
    if return_mse:
        return mse / mean_square_error(0, real), mse
    else:
        return mse / mean_square_error(0, real)


def mean_square_error(pred, real):
    real = tf.cast(real, dtype=tf.float32)
    se = (pred - real)**2
    return tf.reduce_mean(se, axis=0)


if __name__ == '__main__':
    main()
