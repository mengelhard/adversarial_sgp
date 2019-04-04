import numpy as np
import tensorflow as tf

def logdet(matrix):
    chol = tf.cholesky(matrix)
    return 2.0 * tf.reduce_sum(tf.log(tf.real(tf.matrix_diag_part(chol))),reduction_indices=[-1])

def rmult(mat,diag,naxes=3):
    return mat*tf.expand_dims(diag,axis=naxes-2)

def lmult(diag,mat,naxes=3):
    return tf.expand_dims(diag,axis=naxes-1)*mat

def pairwise_distance(x1,x2):
    r1 = tf.reduce_sum(x1**2,axis=2)[:,:,tf.newaxis]
    r2 = tf.reduce_sum(x2**2,axis=2)[:,tf.newaxis,:]
    r12 = tf.matmul(x1,tf.matrix_transpose(x2))
    return r1+r2-2*r12

def scaled_square_dist(x1,x2,sls):
    '''sls is a P by D matrix of squared length scales, 
    where P is the number of particles and D is the number of dimenions'''
    ls = tf.sqrt(sls)[:,tf.newaxis,:]
    return pairwise_distance(x1/ls,x2/ls)

def nlog_sgp_marglik(x,y,m,sls,sfs,noise,obj='vfe'):

    P = tf.shape(m)[0] # number of pseudo-input samples
    y = tf.tile(tf.expand_dims(tf.cast(y,dtype=tf.float32),0),(P,1))
    x = tf.tile(tf.expand_dims(tf.cast(x,dtype=tf.float32),0),(P,1,1))

    D = tf.shape(x)[2]
    X = tf.shape(x)[1]
    M = tf.shape(m)[1]

    logtp = tf.constant(np.log(2.*np.pi),dtype=tf.float32)

    jitter = 1e-6*tf.eye(M,dtype=tf.float32)[tf.newaxis,:,:]

    xm = scaled_square_dist(x,m,sls)
    mm = scaled_square_dist(m,m,sls)

    kxm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*xm)
    kmm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*mm)
    
    kmx = tf.matrix_transpose(kxm)
    kmmi = tf.matrix_inverse(kmm+jitter)

    qmm_diag = tf.reduce_sum(tf.matmul(kxm,kmmi)*kxm,axis=2)

    if obj=='vfe':
    
        gd = noise[:,tf.newaxis]
        gid = 1/gd

        tr = sfs*tf.cast(X,tf.float32)-tf.reduce_sum(qmm_diag,axis=1)
    
        kmx_gi_kxm = tf.matmul(kmx,kxm)/noise[:,tf.newaxis,tf.newaxis]

        giy = gid*y
        kgiy = tf.reduce_sum(kmx*giy[:,tf.newaxis,:],axis=2)

        inner = kmmi+kmx_gi_kxm+jitter

        covd = logdet(inner)+logdet(kmm+jitter)+tf.log(noise)*tf.cast(X,dtype=tf.float32)
    
        t1 = .5*tf.cast(X,tf.float32)*logtp
        t2 = .5*tf.reduce_sum(y*y*gid,axis=1) - .5*tf.reduce_sum(
            tf.reduce_sum(kgiy[:,:,tf.newaxis]*tf.matrix_inverse(inner),axis=1)*kgiy,axis=1)
        t3 = .5*covd
        t4 = .5*tf.div(tr,noise)
    
        return t1+t2+t3+t4

    if obj=='fitc':

        gd = (sfs+noise)[:,tf.newaxis] - qmm_diag
        gid = 1/gd
        
        giy = gid*y
        kgiy = tf.reduce_sum(kmx*giy[:,tf.newaxis,:],axis=2)

        kmx_gi_kxm = tf.matmul(rmult(kmx,gid),kxm)

        inner = kmmi+kmx_gi_kxm+jitter

        covd = logdet(inner)+logdet(kmm+jitter)+tf.reduce_sum(tf.log(gd),axis=1)

        t1 = .5*tf.cast(X,tf.float32)*logtp
        t2 = .5*tf.reduce_sum(y*y*gid,axis=1) - .5*tf.reduce_sum(
            tf.reduce_sum(kgiy[:,:,tf.newaxis]*tf.matrix_inverse(inner),axis=1)*kgiy,axis=1)
        t3 = .5*covd
        
        return t1+t2+t3#-tf.log(tf.reduce_sum(noise)) # how to prevent noise from becoming negative?

    return None

def sgp_pred(x,y,t,m,sls,sfs,noise,obj='vfe'):
    
    P = tf.shape(m)[0] # number of pseudo-input samples
    y = tf.tile(tf.expand_dims(tf.cast(y,dtype=tf.float32),0),(P,1))
    x = tf.tile(tf.expand_dims(tf.cast(x,dtype=tf.float32),0),(P,1,1))
    t = tf.tile(tf.expand_dims(tf.cast(t,dtype=tf.float32),0),(P,1,1))

    D = tf.shape(x)[2]
    X = tf.shape(x)[1]

    logtp = tf.constant(np.log(2.*np.pi),dtype=tf.float32)

    M = tf.shape(m)[1]

    jitter = 1e-6*tf.eye(M,dtype=tf.float32)[tf.newaxis,:,:]

    xm = scaled_square_dist(x,m,sls)
    mm = scaled_square_dist(m,m,sls)
    tm = scaled_square_dist(t,m,sls)

    kxm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*xm)
    kmm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*mm)
    kmx = tf.matrix_transpose(kxm)

    ktm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*tm)

    if obj=='vfe':

        giy = y/noise[:,tf.newaxis]
        kgiy = tf.reduce_sum(kmx*giy[:,tf.newaxis,:],axis=2)
        kmx_gi_kxm = tf.matmul(kmx,kxm)/noise[:,tf.newaxis,tf.newaxis]
        qm = kmm+kmx_gi_kxm+jitter

    if obj=='fitc':

        kmmi = tf.matrix_inverse(kmm+jitter)
        qmm_diag = tf.reduce_sum(tf.matmul(kxm,kmmi)*kxm,axis=2)
        
        gd = (sfs+noise)[:,tf.newaxis] - qmm_diag
        gid = 1/gd

        giy = gid*y
        kgiy = tf.reduce_sum(kmx*giy[:,tf.newaxis,:],axis=2)
        kmx_gi_kxm = tf.matmul(rmult(kmx,gid),kxm)
        qm = kmm+kmx_gi_kxm+jitter

    ph = tf.reduce_sum(tf.matrix_inverse(qm)*kgiy[:,tf.newaxis,:],axis=2)
    mean = tf.reduce_sum(ktm*ph[:,tf.newaxis,:],axis=2)

    return mean
