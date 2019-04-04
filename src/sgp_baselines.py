import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.contrib import slim
from tensorflow.contrib import distributions

from tqdm import tqdm#, tqdm_notebook
import os

from load_data import load_kin40k
from sgp import nlog_sgp_marglik, sgp_pred

#class FLAGS:
#    def __getattr__(self, name):
#        self.__dict__[name] = FLAGS()
#        return self.__dict__[name]
#
#FLAGS = FLAGS()

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('img_path','./img',"""Directory for output figs""")
tf.app.flags.DEFINE_integer('niter',10000,"""Number of iterations""")
tf.app.flags.DEFINE_integer('print_freq',500,"""How often to display results""")
tf.app.flags.DEFINE_float('lr',3e-5,"""Adam learning rate""")
tf.app.flags.DEFINE_string('dataset','kin40k',"""Dataset to Use""")
tf.app.flags.DEFINE_float('val_prc',0.,"""Portion of training data to use for validation""")
tf.app.flags.DEFINE_integer('n_z',100,"""Num inducing inputs""")
tf.app.flags.DEFINE_string('sgp_approx','vfe',"""SGP approximation method""")
tf.app.flags.DEFINE_float('sls',1.,"""Initial GP length scale""")
tf.app.flags.DEFINE_float('sfs',1.,"""Initial GP function scale""")
tf.app.flags.DEFINE_float('noise',1e-3,"""Initial GP noise""")
tf.app.flags.DEFINE_float('lbd',0.,"""Hyperparameter penalty""")
tf.app.flags.DEFINE_integer('seed',1,"""Seed (for clustering)""")
tf.app.flags.DEFINE_integer('hold_time',0,"""How many iterations to hold KHPs""")
tf.app.flags.DEFINE_bool('ard',True,"""Whether to use ARD kernel""")
tf.app.flags.DEFINE_string('initial_z','subset',"""Method for choosing initial inducing inputs""")


def kmeans_mixture_model(d,n_clusters=100,random_state=0):
    """
    Cluster x and return cluster centers and cluster widths (variances)
    """
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=n_clusters,random_state=random_state).fit(d)

    lbls = km.labels_
    cc = km.cluster_centers_

    d_centered = np.array([x - cc[y] for x,y in zip(d,lbls)])
    c_widths = np.array([np.sum(d_centered[lbls==c,:]**2,axis=0)/np.sum(lbls==c) 
        for c in range(n_clusters)])

    weights = np.array([np.sum(lbls==c) for c in range(n_clusters)])
    weights = weights/np.sum(weights)

    mixture_model = {
        'n_components':n_clusters,
        'n_dims':np.shape(d)[1],
        'weights':weights,
        'means':cc,
        'sds':np.sqrt(c_widths)
        }

    return mixture_model

def tsne(points):
    from sklearn.manifold import TSNE
    points_embedded = TSNE(n_components=2).fit_transform(points)
    return points_embedded

def get_mse(pred,real): # check axes
    se = (pred-real)**2
    ms = tf.reduce_mean(se)
    _, var = tf.nn.moments(real, axes=[0])
    return ms, ms/var

def trainplot(val_freq,val_error,train_freq=None,train_error=None,savepath=None):

    fig,ax = plt.subplots(1,1,figsize=(4,4))
    val_iters = val_freq*np.arange(len(val_error))
    ax.plot(val_iters,val_error)
    if train_error is not None:
        train_iters = train_freq*np.arange(len(train_error))
        ax.plot(train_iters,train_error)
    ax.set_ylabel('SMSE')
    ax.set_xlabel('Iteration')

    if savepath is not None:
        plt.savefig(os.path.join(savepath,'training.png'))
    else:
        plt.show()

def z_plot(z,savename=None):

    fig,ax=plt.subplots(1,1,figsize=(4,4))
    ccs = tsne(np.squeeze(z))
    ax.scatter(ccs[:,0],ccs[:,1])

    if savename is not None:
        plt.savefig(savename)
    else:
        plt.show()

def main(argv):

    print(tf.flags.FLAGS.flag_values_dict())

    # Load data
    if FLAGS.dataset == 'kin40k':
        train_x, train_y, val_x, val_y, test_x, test_y = load_kin40k(val_prc=FLAGS.val_prc)

    initial_model = kmeans_mixture_model(train_x,
        n_clusters=FLAGS.n_z,
        random_state=FLAGS.seed)

    if FLAGS.initial_z=='kmeans':
        initial_z = initial_model['means']
    elif FLAGS.initial_z=='subset':
        np.random.seed(FLAGS.seed)
        zidx = np.random.permutation(np.arange(len(train_x)))<FLAGS.n_z
        initial_z = train_x[zidx,:]
    else:
        initial_z = None

    outdir = FLAGS.img_path
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    z_plot(initial_z,savename=os.path.join(outdir,'z_initial.png'))

    if FLAGS.ard:
        sls = tf.expand_dims(tf.Variable(np.repeat(FLAGS.sls,np.shape(train_x)[1]),
            name='sls',dtype=tf.float32),axis=0)
    else:
        sls = tf.Variable(FLAGS.sls,name='sls',dtype=tf.float32)
        sls = tf.tile(sls[tf.newaxis,tf.newaxis],(1,np.shape(train_x)[1]))

    sfs = tf.expand_dims(tf.Variable(FLAGS.sfs,name='sfs',dtype=tf.float32),axis=0)
    noise = tf.expand_dims(tf.Variable(FLAGS.noise,name='noise',dtype=tf.float32),axis=0)

    z = tf.expand_dims(tf.Variable(initial_z,name='z',dtype=tf.float32),axis=0)

    # Get SGP stuff
    sgp_nlogprob = nlog_sgp_marglik(train_x,train_y,z,sls,sfs,noise,obj=FLAGS.sgp_approx)
    penalty = FLAGS.lbd * tf.reduce_mean(noise**2)

    opt = tf.train.AdamOptimizer(FLAGS.lr)
    grad = opt.compute_gradients(sgp_nlogprob+penalty)

    khp_vars = [v for v in tf.global_variables() if v.op.name in ['sls','sfs','noise']]

    train_step = opt.apply_gradients(grad)
    partial_train = opt.apply_gradients([(0.*g,v) if v in khp_vars else (g,v) for g, v in grad])

    pred = sgp_pred(train_x,train_y,test_x,z,sls,sfs,noise,obj=FLAGS.sgp_approx)

    # Get MSE / normalized MSE
    mse, nmse = get_mse(tf.squeeze(pred),tf.constant(test_y,dtype=tf.float32))

    def run_training(sess, niter=FLAGS.niter, 
        printfreq=FLAGS.print_freq, hold_time=FLAGS.hold_time):

        progress = tqdm(range(niter))
        nms_error = []
        ms_error = []

        for i in progress:

            if i<hold_time:
                sgp_nlogprob_,sls_,sfs_,noise_,_ = sess.run(
                    [sgp_nlogprob,sls,sfs,noise,partial_train])
            else:
                sgp_nlogprob_,sls_,sfs_,noise_,_ = sess.run(
                    [sgp_nlogprob,sls,sfs,noise,train_step])

            if i%printfreq==0:
                nmse_,mse_ = sess.run([nmse,mse])
                nms_error.append(nmse_)
                ms_error.append(mse_)

            progress.set_description("nlp=%.0f,lmx=%.3f,lmn=%.3f,s=%.3f,n=%.3f,mse=%.3f,nmse=%.3f"  % (
                sgp_nlogprob_,np.amax(sls_),np.amin(sls_),sfs_,noise_,mse_,nmse_))

        trainplot(FLAGS.print_freq,nms_error,savepath=FLAGS.img_path)
        z_plot(sess.run(z),savename=os.path.join(outdir,'z_final.png'))

    try:
        s.close()
    except NameError:
        pass
    s = tf.InteractiveSession()
    s.run(tf.global_variables_initializer())
    run_training(s,niter=FLAGS.niter)

if __name__ == '__main__':
	tf.app.run()
