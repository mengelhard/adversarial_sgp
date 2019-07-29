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

    tx, ty, vx, vy, mx, my = load_dataset('kin40k', val_prc=.2)

    for num_ind_inputs in [32]:  # , 64, 128, 256, 512, 1024]:

        for gaussian_reference in [True, False]:

            for mw_alpha in [1.]:  # , 10., 100.]:

                print('Starting New Run:')

                # try:

                gloss, val_nmse, avg_test_nmse_, min_test_nmse_ = learn_asgp(
                    tx, ty, vx, vy, mx, my,
                    num_ind_inputs=num_ind_inputs,
                    gaussian_reference=gaussian_reference,
                    mw_alpha=mw_alpha)

                # except Exception as e:
                #    print(e)

                tf.reset_default_graph()


def learn_asgp(tx, ty, vx, vy, mx, my,
               num_ind_inputs=None,
               gaussian_reference=None,
               mw_alpha=None):

    num_components = num_ind_inputs
    num_sgp_samples = 100
    max_steps = 10000
    max_no_improve = 10
    check_val_freq = 20

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

    test_predictions = sgpm.predict(mx)
    val_predictions = sgpm.predict(vx)
    avg_test_prediction = tf.reduce_mean(test_predictions, axis=0)[tf.newaxis, :]
    avg_val_prediction = tf.reduce_mean(val_predictions, axis=0)[tf.newaxis, :]
    avg_test_nmse = normalized_mean_square_error(avg_test_prediction, my)
    avg_val_nmse = normalized_mean_square_error(avg_val_prediction, vy)
    test_nmse = normalized_mean_square_error(test_predictions, my)
    val_nmse = normalized_mean_square_error(val_predictions, vy)
    min_test_nmse = tf.reduce_min(test_nmse, axis=0)
    min_val_nmse = tf.reduce_min(val_nmse, axis=0)

    with tf.Session() as s:

        s.run(tf.global_variables_initializer())

        initial_z_, ref_samples_ = s.run([sgp_gen.samples, sgp_gen.ref_samples])

        for i in range(100):
            _ = s.run(asgp.dtrain_step)

        idx = 0
        no_improve = 0
        best_nmse = 100.

        gloss_all = []
        nmse_all = []

        while (no_improve < max_no_improve) and (idx < max_steps):

            idx += 1

            _ = s.run(asgp.dtrain_step)
            _, gloss_, mask_ = s.run(
                [asgp.gtrain_step, asgp.gloss, sgpm.mask])

            prc_used = np.sum(mask_) / len(mask_)

            if idx % check_val_freq == 0:
                min_val_nmse_ = s.run(min_val_nmse)
                nmse_all.append(min_val_nmse_)
                print('best val nmse: %.3f' % min_val_nmse_)
                if best_nmse < min_val_nmse_:
                    no_improve += 1
                else:
                    best_nmse = min_val_nmse_
                    no_improve = 0

        avg_test_nmse_, min_test_nmse_ = s.run([avg_test_nmse, min_test_nmse])
        print('avg test nmse: %.3f' % avg_test_nmse_)
        print('best test nmse: %.3f' % min_test_nmse_)

    return gloss_all, nmse_all, avg_test_nmse_, min_test_nmse_


def plot_kin_initial(tx, ty, ref_samples, z):

    return None


def plot_kin_final(tx, ty, ref_samples, z):

    return None


def normalized_mean_square_error(pred, real, return_mse=False):
    mse = mean_square_error(pred, real)
    if return_mse:
        return mse / mean_square_error(0, real), mse
    else:
        return mse / mean_square_error(0, real)


def mean_square_error(pred, real):
    real = tf.cast(real, dtype=tf.float32)[tf.newaxis, :]
    se = (pred - real)**2
    return tf.reduce_mean(se, axis=1)


if __name__ == '__main__':
    main()
