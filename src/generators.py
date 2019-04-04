
import numpy as np
import tensorflow as tf

def generator_simple(initial_model,tau_initial=1e-4,nn_depth=3,reuse=False):
    """
    Use random noise 'eps' to sample from mixture model
    """
    
    #eps_unif = tf.random_uniform((1,
    #    initial_model['n_components']),dtype=tf.float32)

    eps_gauss = tf.random_normal((FLAGS.batch_size,
        initial_model['n_dims']),dtype=tf.float32)

    with tf.variable_scope("generator", reuse=reuse) as scope:
        
        means = slim.model_variable('means',
                                    shape=np.shape(initial_model['means']),
                                    initializer=tf.constant_initializer(initial_model['means'])
                                   )
        
        sds = slim.model_variable('sds',
                                  shape=np.shape(initial_model['sds']),
                                  initializer=tf.constant_initializer(initial_model['sds'])
                                 )
        
        w = slim.model_variable('weightlogits',
                                shape=(initial_model['n_components']),
                                initializer=tf.zeros_initializer()
                               )
                                      
        tau = slim.model_variable('tau',
                                  shape=np.shape(tau_initial),
                                  initializer=tf.constant_initializer(tau_initial)
                                 )
        
        weights = gumbel_softmax_sample(w[tf.newaxis,:],
            temperature=tau,num_samples=FLAGS.batch_size)
        
        y = tf.reduce_sum(weights[:,:,tf.newaxis] * means[tf.newaxis,:,:], axis=1)
        y += tf.reduce_sum(weights[:,:,tf.newaxis] * sds[tf.newaxis,:,:],axis=1) * eps_gauss
        
        return y, weights

def generator_direct(initial_model,tau=1e-4,nn_depth=3,reuse=False):
    """
    Use random noise 'eps' to sample from mixture model
    """

    eps_unif = tf.random_uniform((FLAGS.batch_size,
        initial_model['n_components']),dtype=tf.float32)

    eps_gauss = tf.random_normal((FLAGS.batch_size,
        initial_model['n_dims']),dtype=tf.float32)

    with tf.variable_scope("generator", reuse=reuse) as scope:
        
        means = slim.model_variable('means',
                                    shape=np.shape(initial_model['means']),
                                    initializer=tf.constant_initializer(initial_model['means'])
                                   )
        
        sds = slim.model_variable('sds',
                                  shape=np.shape(initial_model['sds']),
                                  initializer=tf.constant_initializer(initial_model['sds'])
                                 )

        net = eps_unif

        for i in range(nn_depth):
            net = slim.fully_connected(net,initial_model['n_components'],
                activation_fn=tf.nn.elu,scope='fc_%d' % (i+1))
        
        weights = slim.softmax(net/tau)

        y = tf.reduce_sum(weights[:,:,tf.newaxis] * means[tf.newaxis,:,:], axis=1)
        y += tf.reduce_sum(weights[:,:,tf.newaxis] * sds[tf.newaxis,:,:],axis=1) * eps_gauss
        
        return y, weights

def generator_gumbel(initial_model,tau_initial=1e-4,nn_depth=3,reuse=False):
    """
    Use random noise 'eps' to sample from gumbel softmax, then sample mixture
    """

    #eps_unif = tf.random_uniform((FLAGS.batch_size,
    #    initial_model['n_components']),dtype=tf.float32)

    eps_unif = tf.random_uniform([1,initial_model['n_components']],dtype=tf.float32)
    
    # switch this to do less?
    # eps_unif = tf.tile(tf.random_uniform(initial_model['n_components'],
    # dtype=tf.float32)[tf.newaxis,:],
    #    batch_size,[batch_size,1])

    eps_gauss = tf.random_normal((FLAGS.batch_size,
        initial_model['n_dims']),dtype=tf.float32)

    with tf.variable_scope("generator", reuse=reuse) as scope:

        #tau = slim.model_variable('tau',
        #    shape=np.shape(tau_initial),
        #    initializer=tf.constant_initializer(tau_initial)
        #    )
        
        #for now, let's try fixed tau
        
        tau = tau_initial
        
        means = slim.model_variable('means',
            shape=np.shape(initial_model['means']),
            initializer=tf.constant_initializer(initial_model['means'])
            )
        
        sds = slim.model_variable('sds',
            shape=np.shape(initial_model['sds']),
            initializer=tf.constant_initializer(initial_model['sds'])
            )
        
        with slim.arg_scope([slim.fully_connected], activation_fn=lrelu):
            net = slim.fully_connected(eps_unif, 256, scope='fc_0')
            
            for i in range(nn_depth):
                dnet = slim.fully_connected(net, 256, scope='fc_%d_r0' % (i+1))
                net += slim.fully_connected(dnet, 256, activation_fn=None, scope='fc_%d_r1' % (i+1),
                    weights_initializer=tf.constant_initializer(0.))
                net = lrelu(net)

        T = slim.fully_connected(net, initial_model['n_components'], activation_fn=None, scope='T',
            weights_initializer=tf.constant_initializer(0.))

        #weights = tf.nn.softmax((tf.nn.log_softmax(T,axis=1) + gumbel)/tau,axis=1)

        weights = gumbel_softmax_sample(tf.nn.log_softmax(T,axis=1),
            temperature=tau,num_samples=FLAGS.batch_size)
                
        # dimension should be: (batch_size, n_components, dimension)
        y = tf.reduce_sum(weights[:,:,tf.newaxis] * means[tf.newaxis,:,:], axis=1)
        y += tf.reduce_sum(weights[:,:,tf.newaxis] * sds[tf.newaxis,:,:],axis=1) * eps_gauss
                
        return y, weights