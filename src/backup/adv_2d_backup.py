import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.contrib import slim
from tensorflow.contrib import distributions

from tqdm import tqdm
import os

BATCH_SIZE = 1000
D_LR = 1e-4
G_LR = 1e-4

def main():

    ### specify mixture model to be learned
    # this is the target distribution

    target_model = {'n_components':3,
                     'n_dims':2,
                     'weights':[.2,.2,.6],
                     'means':[[-.6,1.6],[-1.2,-1.3],[1.,.6]],
                     'sds':[[.3,.1],[.2,.2],[.2,.6]]
                    }

    initial_model = {'n_components':3,
                     'n_dims':2,
                     'weights':[1./3,1./3,1./3],
                     'means':[[-1.,1.],[-1.,-1.],[1.,0.]],
                     'sds':[[.6,.6],[.6,.6],[.6,.6]]
                    }

    #target_model = initial_model

    def mixture_pdf(model):
        dist = distributions.MixtureSameFamily(
            mixture_distribution=distributions.Categorical(
                probs=model['weights']),
            components_distribution=distributions.MultivariateNormalDiag(
                loc=model['means'],       # One for each component.
                scale_diag=model['sds']))  # And same here.
        return dist

    mp = mixture_pdf(target_model)
    mixlogprob = mp.log_prob
    mixsample = mp.sample

    # specify the generator that will learn this mixture model

    def generator(eps_unif,eps_gauss,tau=.01,n_layers=3,reuse=False):
        """
        Use random noise 'eps' to sample from mixture model
        """
        with tf.variable_scope("generator", reuse=reuse) as scope:
            
            means = slim.model_variable('means',
                                        shape=np.shape(initial_model['means']),
                                        initializer=tf.constant_initializer(initial_model['means'])
                                       )
            
            sds = slim.model_variable('sds',
                                      shape=np.shape(initial_model['sds']),
                                      initializer=tf.constant_initializer(initial_model['sds'])
                                     )
            
            slim.stack(eps_unif, slim.fully_connected, 
                       [10,10,initial_model['n_components']], 
                       activation_fn=tf.nn.elu, 
                       scope='fc'
                      )
            
            weights = slim.softmax(eps_unif/tau)
            
            y = tf.reduce_sum(weights[:,:,tf.newaxis] * means[tf.newaxis,:,:], axis=1)
            y += tf.reduce_sum(weights[:,:,tf.newaxis] * sds[tf.newaxis,:,:],axis=1) * eps_gauss
            
            return y

    eps_unif = tf.random_uniform((BATCH_SIZE,initial_model['n_components']),dtype=tf.float32)
    eps_gauss = tf.random_normal((BATCH_SIZE,initial_model['n_dims']),dtype=tf.float32)

    y = generator(eps_unif,eps_gauss)

    # specify the reference distribution. we need to both evaluate and sample

    ref = distributions.MultivariateNormalDiag(
        loc=[0.,0.],
        scale_diag=[2.5,2.5]
    )

    reflogprob = ref.log_prob
    refsample = ref.sample(sample_shape=BATCH_SIZE)

    # visualize the target model

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        initial_samples = s.run(mixsample(sample_shape=BATCH_SIZE))

    # specify the adversary, which will learn the likelihood ratio
    # between generator samples and reference samples

    def lrelu(x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak*x)

    def adversary(y, reuse=False):
        with tf.variable_scope("adversary", reuse=reuse) as scope:
            with slim.arg_scope([slim.fully_connected], activation_fn=lrelu):
                net = slim.fully_connected(y, 256, scope='fc_0')

                for i in range(5):
                    dnet = slim.fully_connected(net, 256, scope='fc_%d_r0' % (i+1))
                    net += slim.fully_connected(dnet, 256, activation_fn=None, scope='fc_%d_r1' % (i+1),
                                                weights_initializer=tf.constant_initializer(0.))
                    net = lrelu(net) 

            T = slim.fully_connected(net, 1, activation_fn=None, scope='T',
                                    weights_initializer=tf.constant_initializer(0.))
            T = tf.squeeze(T, [1])
            return T

    # prepare for run

    dref = adversary(refsample)
    dy = adversary(y,reuse=True)

    dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "adversary")
    gvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")

    dloss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=dy, labels=tf.ones_like(dy))
        + tf.nn.sigmoid_cross_entropy_with_logits(logits=dref, labels=tf.zeros_like(dref))
    )

    gloss = tf.reduce_sum(reflogprob(y) - mixlogprob(y) + dy, axis=0)

    dtrain_step = tf.train.AdamOptimizer(D_LR).minimize(dloss,var_list=dvars)
    gtrain_step = tf.train.AdamOptimizer(G_LR).minimize(gloss,var_list=gvars)

    outdir = './out_mix_2d'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    def run_training(sess, niter=10001, ninitial=100):
        
        # consider a burn-in period for the discriminator before going to the generator
        
        burn_in = tqdm(range(ninitial))
        
        for i in burn_in:
            
            y_,refsample_,dref_,dy_,dloss_,_ = s.run([y,refsample,dref,dy,dloss,dtrain_step])
            
            burn_in.set_description("dloss=%.3f"  % (dloss_))

        progress = tqdm(range(niter))

        for i in progress:
                  
            y_,refsample_,dref_,dy_,dloss_,_ = s.run([y,refsample,dref,dy,dloss,dtrain_step])
            _ = s.run([gtrain_step])
            
            #print(np.shape(y_),np.shape(refsample_))
                
            progress.set_description("dloss=%.3f"  % (dloss_))

            dispfreq = 2000
            
            if i%dispfreq == 0:
                fig,ax=plt.subplots(1,4,figsize=(16,4))
                
                ax[0].scatter(initial_samples[:,0],initial_samples[:,1],alpha=.3)
                ax[0].set_xlim([-4,4])
                ax[0].set_ylim([-4,4])
                ax[0].set_title('Target Samples')
                
                ax[1].scatter(y_[:,0],y_[:,1],alpha=.3)
                ax[2].scatter(refsample_[:,0],refsample_[:,1],alpha=.3)
                ax[1].set_xlim([-4,4])
                ax[1].set_ylim([-4,4])
                ax[2].set_xlim([-4,4])
                ax[2].set_ylim([-4,4])
                
                ax[1].set_title('Generated Samples')
                ax[2].set_title('Reference Samples')
                ax[3].scatter(refsample_[:,0],refsample_[:,1],alpha=.3,c=dref_,cmap='plasma')
                
                ax[3].set_xlim([-4,4])
                ax[3].set_ylim([-4,4])
                ax[3].set_title('Discriminator on Ref')

                plt.savefig('out_mix_2d/iter_%i.png' % i)

    try:
        s.close()
    except NameError:
        pass
    s = tf.InteractiveSession()
    s.run(tf.global_variables_initializer())
    run_training(s,niter=200000)

if __name__ == '__main__':
    main()