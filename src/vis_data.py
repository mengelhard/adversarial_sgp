import numpy as np
import matplotlib.pyplot as plt
import os


class training_summary:

    def __init__(self, metrics):

        self.metrics = {x: [] for x in metrics}

    def add_point(self, metric, point):

        self.metrics[metric].append(point)

    def plot_metrics(self, ax, metrics):

        for metric in metrics:

            ax.plot(*np.array(self.metrics[metric]).T)

        ax.legend(metrics)
        ax.set_xlabel('Iteration')

        return ax

    def get_metric(self, metric):

        return np.array(self.metrics[metric])


def vis_gen(dataset, *args, **kwargs):
    """Visualize Generator (during training)"""

    if dataset == 'kin40k':
        return vis_gen_kin40k(*args, **kwargs)
    elif dataset == 'abalone':
        return vis_gen_abalone(*args, **kwargs)
    elif dataset == 'boston':
        return vis_gen_boston(*args, **kwargs)
    elif dataset == 'sarcos':
        return vis_gen_sarcos(*args, **kwargs)
    elif dataset == 'mcycle':
        return vis_gen_mcycle(*args, **kwargs)
    else:
        return None


def vis_gen_kin40k(*args, **kwargs):

    return None


def vis_gen_abalone(*args, **kwargs):

    return None


def vis_gen_boston(*args, **kwargs):

    return None


def vis_gen_sarcos(*args, **kwargs):

    return None


def vis_gen_mcycle(ax, z, x, y, *args, **kwargs):

    # print('Shape of z:')
    # print(np.shape(z))

    x = np.squeeze(x)
    miny = np.amin(y) - .02

    ax.plot(x, y, 'x')

    for i, z_pts in enumerate(z):
        z_pts = np.squeeze(z_pts)
        ax.plot(z_pts, z_pts * 0 + miny - .02 * i, 'x')

    return ax
