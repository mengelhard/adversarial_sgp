import numpy as np
import tensorflow as tf
from scipy.special import gammaln
import matplotlib.pyplot as plt
import itertools

#def pairwise_distance_np(x1,x2):
#	return np.transpose((x1[np.newaxis,:]-x2[:,np.newaxis])**2)

def logdet(matrix):
	chol = tf.cholesky(matrix)
	return 2.0 * tf.reduce_sum(tf.log(tf.real(tf.matrix_diag_part(chol))),reduction_indices=[-1])

def pairwise_distance(x1,x2):
	r1 = tf.reduce_sum(x1**2,axis=2)[:,:,tf.newaxis]
	r2 = tf.reduce_sum(x2**2,axis=2)[:,tf.newaxis,:]
	r12 = tf.matmul(x1,tf.matrix_transpose(x2))
	return r1+r2-2*r12

def pairwise_distance_np(x1,x2):
	r1 = np.sum(x1**2,axis=1)[:,np.newaxis]
	r2 = np.sum(x2**2,axis=1)[np.newaxis,:]
	r12 = x1@(x2.T)
	return r1+r2-2*r12

def scaled_square_dist(x1,x2,sls):
	'''sls is a P by D matrix of squared length scales, 
	where P is the number of particles and D is the number of dimenions'''
	ls = tf.sqrt(sls)[:,tf.newaxis,:]
	return pairwise_distance(x1/ls,x2/ls)

def scaled_square_dist_np(x1,x2,sls):
	'''sls is a P by D matrix of squared length scales, 
	where P is the number of particles and D is the number of dimenions'''
	ls = np.sqrt(sls)[np.newaxis,:]
	x1 = x1/ls
	x2 = x2/ls
	r1 = np.sum(x1**2,axis=1)[:,np.newaxis]
	r2 = np.sum(x2**2,axis=1)[np.newaxis,:]
	r12 = x1@(x2.T)
	return r1+r2-2*r12

def ssd_ard(z,D):
	m = z[:,D+2:]
	m = tf.reshape(m,[tf.shape(m)[0],-1,D])
	centers = tf.reduce_mean(m,axis=1)
	sumsqdist = tf.reduce_sum((m-centers[:,tf.newaxis,:])**2,axis=[1,2])
	return sumsqdist[:,tf.newaxis]

def ssd(z,D):
	m = z[:,3:]
	m = tf.reshape(m,[tf.shape(m)[0],-1,D])
	centers = tf.reduce_mean(m,axis=1)
	sumsqdist = tf.reduce_sum((m-centers[:,tf.newaxis,:])**2,axis=[1,2])
	return sumsqdist[:,tf.newaxis]

def sample_normal(mean,cov,n_samples):
	l = np.linalg.cholesky(cov)
	samples = []
	dim = len(mean)
	for i in range(n_samples):
		samples.append(mean + l@np.random.normal(0,1,dim))
	return samples

def loggamma(x,k,theta):
	return -k*np.log(theta)-gammaln(k)+(k-1)*tf.log(x)-x/theta

def lognormal(x,mean,var,logtp):
	return ((x-mean)**2)/(2*var)-.5*np.log(2*3.14*var)

def initialize_pips(boundaries,rpd,pprpd,ppr,jitter=1e-2):
	'''Initialize pseudo-inputs.
	boundaries is a list of intervals, one for each dimension
	rpd is the number of regions per dimension
	pprpd is the number of pseudo-inputs per rpd
	ppr is the number of particles per region'''
	rb = [intervals(b[0],b[1],rpd) for b in boundaries]
	regions = list(itertools.product(*rb))
	ip = [rpoints(r,pprpd) for r in regions]
	ip = np.tile(np.squeeze(np.array(ip).reshape(len(ip),-1,1)),(ppr,1))
	return ip + jitter*np.random.randn(*np.shape(ip))

def intervals(low,high,num):
	sz = (high-low)/num
	return [[low+sz*i,low+sz*(i+1)] for i in range(num)]

def rpoints(r,pprpd):
	coords = np.meshgrid(*[np.linspace(x[0],x[1],pprpd) for x in r])
	return np.hstack([x.reshape(-1,1) for x in coords])

def sgp_samples(x,y,m,t,sls,sfs,noise,n_samples=1):

	'''Sample from sparse GP
	x = inputs
	y = data
	m = pseudo-inputs
	t = gridpoints for the sample
	sls = length scale
	sfs = amplitude
	noise = noise
	n_samples = number of samples to draw'''

	noise = np.abs(noise)

	kxm = sfs*np.exp(-.5*pairwise_distance_np(x,m)/sls)
	kmm = sfs*np.exp(-.5*pairwise_distance_np(m,m)/sls)
	kmx = kxm.T

	kmm_inv = np.linalg.inv(kmm)

	#gam_diag = [gp_params[1]-kmx[:,i].T@kmm_inv@kmx[:,i] for i in range(len(x))]
	gam_diag = sfs - np.sum(np.matmul(kxm,kmm_inv)*kxm,axis=1)
	gam = np.diag(gam_diag)
	#gameyeinv = np.diag([1/(g+sigsq) for g in gam_diag])
	gameyeinv = np.diag(1/(gam_diag+noise))

	qm = kmm + kmx@gameyeinv@kxm
	qm_inv = np.linalg.inv(qm)

	ktm = sfs*np.exp(-.5*pairwise_distance_np(t,m)/sls)
	ktt = sfs*np.exp(-.5*pairwise_distance_np(t,t)/sls)
	kmt = ktm.T

	mean = ktm@qm_inv@kmx@gameyeinv@y
	#cov = ktt - ktm@(kmm_inv - qm_inv)@kmt + noise*np.identity(len(t))
	cov = ktt - ktm@(kmm_inv - qm_inv)@kmt + .000001*np.identity(len(t))

	samples = sample_normal(mean,cov,n_samples)

	return samples

def sgp_pred(x,y,t,z):

	D = np.shape(x)[1]
	
	sls = z[0]
	sfs = z[1]
	noise = z[2]

	m = z[3:]
	m = np.reshape(m,[-1,D]) # unflatten pseudo-inputs

	kxm = sfs*np.exp(-.5*pairwise_distance_np(x,m)/sls)
	kmm = sfs*np.exp(-.5*pairwise_distance_np(m,m)/sls)
	kmx = kxm.T

	jitter = 1e-6*np.eye(len(m))

	kmm_inv = np.linalg.inv(kmm+jitter)

	#gam_diag = [gp_params[1]-kmx[:,i].T@kmm_inv@kmx[:,i] for i in range(len(x))]
	gam_diag = sfs - np.sum(np.matmul(kxm,kmm_inv)*kxm,axis=1)
	gam = np.diag(gam_diag)
	#gameyeinv = np.diag([1/(g+sigsq) for g in gam_diag])
	gameyeinv = np.diag(1/(gam_diag+noise))

	qm = kmm + kmx@gameyeinv@kxm
	qm_inv = np.linalg.inv(qm+jitter)

	ktm = sfs*np.exp(-.5*pairwise_distance_np(t,m)/sls)
	ktt = sfs*np.exp(-.5*pairwise_distance_np(t,t)/sls)
	kmt = ktm.T

	mean = ktm@qm_inv@kmx@gameyeinv@y
	cov = ktt - ktm@(kmm_inv - qm_inv)@kmt + noise*np.identity(len(t))

	return mean, cov

def vfe_pred(x,y,t,z):

	z = tf.transpose(z)

	P = tf.shape(z)[1] # number of particles

	y = tf.tile(tf.expand_dims(y,0),(P,1))
	x = tf.tile(tf.expand_dims(x,0),(P,1,1))
	t = tf.tile(tf.expand_dims(t,0),(P,1,1))
	
	D = tf.shape(x)[2]
	X = tf.shape(x)[1]
	
	sls = z[0]
	sfs = z[1]
	noise = z[2]

	m = tf.transpose(z[3:])
	m = tf.reshape(m,[P,-1,D]) # unflatten pseudo-inputs

	M = tf.shape(m)[1]

	jitter = 1e-6*tf.eye(M,dtype=tf.float64)[tf.newaxis,:,:]

	xm = pairwise_distance(x,m)
	mm = pairwise_distance(m,m)

	kxm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*xm/sls[:,tf.newaxis,tf.newaxis])
	kmm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*mm/sls[:,tf.newaxis,tf.newaxis])
	kmx = tf.matrix_transpose(kxm)

	tm = pairwise_distance(t,m)

	ktm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*tm/sls[:,tf.newaxis,tf.newaxis])

	giy = y/noise[:,tf.newaxis]
	kgiy = tf.reduce_sum(kmx*giy[:,tf.newaxis,:],axis=2)
	kmx_gi_kxm = tf.matmul(kmx,kxm)/noise[:,tf.newaxis,tf.newaxis]
	qm = kmm+kmx_gi_kxm+jitter

	ph = tf.reduce_sum(tf.matrix_inverse(qm)*kgiy[:,tf.newaxis,:],axis=2)
	mean = tf.reduce_sum(ktm*ph[:,tf.newaxis,:],axis=2)

	return mean

def vfe_pred_ard(x,y,t,z):

	z = tf.transpose(z)

	P = tf.shape(z)[1] # number of particles
	y = tf.tile(tf.expand_dims(y,0),(P,1))
	x = tf.tile(tf.expand_dims(x,0),(P,1,1))
	t = tf.tile(tf.expand_dims(t,0),(P,1,1))

	D = tf.shape(x)[2]
	X = tf.shape(x)[1]

	sls = tf.transpose(z[:D])
	sfs = z[D]
	noise = z[D+1]

	m = tf.transpose(z[D+2:])
	m = tf.reshape(m,[tf.shape(m)[0],-1,D]) # unflatten pseudo-inputs
	m = m + 1e-8*tf.random_normal(tf.shape(m),dtype=tf.float64)

	M = tf.shape(m)[1]

	jitter = 1e-6*tf.eye(M,dtype=tf.float64)[tf.newaxis,:,:]

	xm = scaled_square_dist(x,m,sls)
	mm = scaled_square_dist(m,m,sls)

	kxm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*xm)
	kmm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*mm)
	kmx = tf.matrix_transpose(kxm)

	tm = scaled_square_dist(t,m,sls)

	ktm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*tm)

	giy = y/noise[:,tf.newaxis]
	kgiy = tf.reduce_sum(kmx*giy[:,tf.newaxis,:],axis=2)
	kmx_gi_kxm = tf.matmul(kmx,kxm)/noise[:,tf.newaxis,tf.newaxis]
	qm = kmm+kmx_gi_kxm+jitter

	ph = tf.reduce_sum(tf.matrix_inverse(qm)*kgiy[:,tf.newaxis,:],axis=2)
	mean = tf.reduce_sum(ktm*ph[:,tf.newaxis,:],axis=2)

	return mean

def fitc_pred(x,y,t,z):

	z = tf.transpose(z)

	P = tf.shape(z)[1] # number of particles

	y = tf.tile(tf.expand_dims(y,0),(P,1))
	x = tf.tile(tf.expand_dims(x,0),(P,1,1))
	t = tf.tile(tf.expand_dims(t,0),(P,1,1))
	
	D = tf.shape(x)[2]
	X = tf.shape(x)[1]
	T = tf.shape(t)[1]
	
	sls = z[0]
	sfs = z[1]
	noise = z[2]

	logtp = tf.constant(np.log(2.*np.pi),dtype=tf.float64)

	m = tf.transpose(z[3:])
	m = tf.reshape(m,[P,-1,D]) # unflatten pseudo-inputs

	M = tf.shape(m)[1]

	jitter = 1e-6*tf.eye(M,dtype=tf.float64)[tf.newaxis,:,:]

	xm = pairwise_distance(x,m)
	mm = pairwise_distance(m,m)

	kxm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*xm/sls[:,tf.newaxis,tf.newaxis])
	kmm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*mm/sls[:,tf.newaxis,tf.newaxis])
	kmx = tf.matrix_transpose(kxm)

	tm = pairwise_distance(t,m)

	ktm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*tm/sls[:,tf.newaxis,tf.newaxis])
	
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

def sgp_pred_ard(x,y,t,z):

	D = np.shape(x)[1]
	
	sls = z[:D]
	sfs = z[D]
	noise = z[D+1]

	m = z[D+2:]
	m = np.reshape(m,[-1,D]) # unflatten pseudo-inputs

	#kxm = sfs*np.exp(-.5*pairwise_distance_np(x,m)/sls)
	#kmm = sfs*np.exp(-.5*pairwise_distance_np(m,m)/sls)
	kxm = sfs*np.exp(-.5*scaled_square_dist_np(x,m,sls))
	kmm = sfs*np.exp(-.5*scaled_square_dist_np(m,m,sls))
	kmx = kxm.T

	jitter = 1e-6*np.eye(len(m))

	kmm_inv = np.linalg.inv(kmm+jitter)

	#gam_diag = [gp_params[1]-kmx[:,i].T@kmm_inv@kmx[:,i] for i in range(len(x))]
	gam_diag = sfs - np.sum(np.matmul(kxm,kmm_inv)*kxm,axis=1)
	gam = np.diag(gam_diag)
	#gameyeinv = np.diag([1/(g+sigsq) for g in gam_diag])
	gameyeinv = np.diag(1/(gam_diag+noise))

	qm = kmm + kmx@gameyeinv@kxm
	qm_inv = np.linalg.inv(qm+jitter)

	#ktm = sfs*np.exp(-.5*pairwise_distance_np(t,m)/sls)
	#ktt = sfs*np.exp(-.5*pairwise_distance_np(t,t)/sls)
	ktm = sfs*np.exp(-.5*scaled_square_dist_np(t,m,sls))
	ktt = sfs*np.exp(-.5*scaled_square_dist_np(t,t,sls))
	kmt = ktm.T

	mean = ktm@qm_inv@kmx@gameyeinv@y
	cov = ktt - ktm@(kmm_inv - qm_inv)@kmt + noise*np.identity(len(t))

	return mean, cov

def nlog_gammaprior(z,alpha,beta):
	logg = alpha*np.log(beta)-gammaln(alpha)+(alpha-1)*tf.log(z)-beta*z
	return -tf.reduce_sum(logg,axis=1)

def nlog_fitc(x,y,z):

	z = tf.transpose(z)

	P = tf.shape(z)[1] # number of particles
	y = tf.tile(tf.expand_dims(y,0),(P,1))
	x = tf.tile(tf.expand_dims(x,0),(P,1,1))
	
	D = tf.shape(x)[2]
	X = tf.shape(x)[1]
	
	sls = z[0]
	sfs = z[1]
	noise = z[2]

	logtp = tf.constant(np.log(2.*np.pi),dtype=tf.float64)

	m = tf.transpose(z[3:])
	m = tf.reshape(m,[P,-1,D]) # unflatten pseudo-inputs
	m = m + 1e-6*tf.random_normal(tf.shape(m),dtype=tf.float64)

	M = tf.shape(m)[1]

	jitter = 1e-6*tf.eye(M,dtype=tf.float64)[tf.newaxis,:,:]

	xm = pairwise_distance(x,m)
	mm = pairwise_distance(m,m)

	kxm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*xm/sls[:,tf.newaxis,tf.newaxis])
	kmm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*mm/sls[:,tf.newaxis,tf.newaxis])
	kmx = tf.matrix_transpose(kxm)
	
	kmmi = tf.matrix_inverse(kmm+jitter)

	qmm_diag = tf.reduce_sum(tf.matmul(kxm,kmmi)*kxm,axis=2)
	
	gd = (sfs+noise)[:,tf.newaxis] - qmm_diag
	gid = 1/gd
	
	giy = gid*y
	kgiy = tf.reduce_sum(kmx*giy[:,tf.newaxis,:],axis=2)

	kmx_gi_kxm = tf.matmul(rmult(kmx,gid),kxm)

	inner = kmmi+kmx_gi_kxm+jitter

	covd = tf.linalg.logdet(inner)+tf.linalg.logdet(kmm+jitter)+tf.reduce_sum(tf.log(gd),axis=1)

	t1 = .5*tf.cast(X,tf.float64)*logtp
	t2 = .5*tf.reduce_sum(y*y*gid,axis=1) - .5*tf.reduce_sum(tf.reduce_sum(kgiy[:,:,tf.newaxis]*tf.matrix_inverse(inner),axis=1)*kgiy,axis=1)
	t3 = .5*covd
	
	return t1+t2+t3

def nlog_fitc_ard(x,y,z):

	z = tf.transpose(z)

	P = tf.shape(z)[1] # number of particles
	ym = tf.tile(tf.expand_dims(tf.expand_dims(y,0),2),(P,1,1))
	x = tf.tile(tf.expand_dims(x,0),(P,1,1))
	
	D = tf.shape(x)[2]
	X = tf.shape(x)[1]
	
	sls = tf.transpose(z[:D])
	sfs = z[D]
	noise = z[D+1]

	logtp = tf.constant(np.log(2.*np.pi),dtype=tf.float64)

	m = tf.transpose(z[D+2:])
	m = tf.reshape(m,[tf.shape(m)[0],-1,D]) # unflatten pseudo-inputs
	m = m + 1e-6*tf.random_normal(tf.shape(m),dtype=tf.float64)

	M = tf.shape(m)[1]

	jitter = 1e-6*tf.eye(M,dtype=tf.float64)[tf.newaxis,:,:]
	
	xm = scaled_square_dist(x,m,sls)
	mm = scaled_square_dist(m,m,sls)

	kxm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*xm)
	kmm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*mm)
	kmx = tf.matrix_transpose(kxm)
	
	kmmi = tf.linalg.inv(kmm+jitter)
	
	qmm = tf.matmul(tf.matmul(kxm,kmmi),kmx)
	
	gd = (sfs+noise)[:,tf.newaxis] - tf.matrix_diag_part(qmm)
	gid = 1/gd
	g = tf.matrix_diag(gd)
	gi = tf.matrix_diag(gid)
	
	cov = qmm+g
	kmx_gi_kxm = tf.matmul(rmult(kmx,gid),kxm)
	covi = gi - lmult(gid,tf.matmul(kxm,rmult(tf.matmul(tf.matrix_inverse(kmmi+kmx_gi_kxm+jitter),kmx),gid)))
	covd = tf.linalg.logdet(kmmi+kmx_gi_kxm+jitter)+tf.linalg.logdet(kmm+jitter)+tf.reduce_sum(tf.log(gd),axis=1)
	
	t1 = .5*tf.cast(X,tf.float64)*logtp
	t2 = .5*tf.squeeze(tf.matmul(tf.matmul(tf.matrix_transpose(ym),covi),ym))
	t3 = .5*covd
	
	return t1+t2+t3

def nlog_vfe(x,y,z):

	z = tf.transpose(z)

	P = tf.shape(z)[1] # number of particles
	y = tf.tile(tf.expand_dims(y,0),(P,1))
	x = tf.tile(tf.expand_dims(x,0),(P,1,1))

	D = tf.shape(x)[2]
	X = tf.shape(x)[1]
	
	sls = z[0]
	sfs = z[1]
	noise = z[2]

	logtp = tf.constant(np.log(2.*np.pi),dtype=tf.float64)

	m = tf.transpose(z[3:])
	m = tf.reshape(m,[P,-1,D]) # unflatten pseudo-inputs
	m = m + 1e-6*tf.random_normal(tf.shape(m),dtype=tf.float64)

	M = tf.shape(m)[1]

	jitter = 1e-6*tf.eye(M,dtype=tf.float64)[tf.newaxis,:,:]

	xm = pairwise_distance(x,m)
	mm = pairwise_distance(m,m)

	kxm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*xm/sls[:,tf.newaxis,tf.newaxis])
	kmm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*mm/sls[:,tf.newaxis,tf.newaxis])
	
	kmx = tf.matrix_transpose(kxm)
	kmmi = tf.matrix_inverse(kmm+jitter)

	qmm_diag = tf.reduce_sum(tf.matmul(kxm,kmmi)*kxm,axis=2)
	
	gd = noise[:,tf.newaxis]
	gid = 1/gd

	tr = sfs*tf.cast(X,tf.float64)-tf.reduce_sum(qmm_diag,axis=1)
	
	kmx_gi_kxm = tf.matmul(kmx,kxm)/noise[:,tf.newaxis,tf.newaxis]

	giy = gid*y
	kgiy = tf.reduce_sum(kmx*giy[:,tf.newaxis,:],axis=2)

	inner = kmmi+kmx_gi_kxm+jitter

	covd = logdet(inner)+logdet(kmm+jitter)+tf.log(noise)*tf.cast(X,dtype=tf.float64)#tf.reduce_sum(tf.log(gd),axis=1)
	
	t1 = .5*tf.cast(X,tf.float64)*logtp
	t2 = .5*tf.reduce_sum(y*y*gid,axis=1) - .5*tf.reduce_sum(tf.reduce_sum(kgiy[:,:,tf.newaxis]*tf.matrix_inverse(inner),axis=1)*kgiy,axis=1)
	t3 = .5*covd
	t4 = .5*tf.div(tr,noise)
	
	return t1+t2+t3+t4

def nlog_vfe_ard(x,y,z):

	z = tf.transpose(z)

	P = tf.shape(z)[1] # number of particles
	y = tf.tile(tf.expand_dims(y,0),(P,1))
	x = tf.tile(tf.expand_dims(x,0),(P,1,1))

	D = tf.shape(x)[2]
	X = tf.shape(x)[1]

	sls = tf.transpose(z[:D])
	sfs = z[D]
	noise = z[D+1]

	logtp = tf.constant(np.log(2.*np.pi),dtype=tf.float64)

	m = tf.transpose(z[D+2:])
	m = tf.reshape(m,[tf.shape(m)[0],-1,D]) # unflatten pseudo-inputs
	m = m + 1e-6*tf.random_normal(tf.shape(m),dtype=tf.float64)

	M = tf.shape(m)[1]

	jitter = 1e-6*tf.eye(M,dtype=tf.float64)[tf.newaxis,:,:]

	xm = scaled_square_dist(x,m,sls)
	mm = scaled_square_dist(m,m,sls)

	kxm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*xm)
	kmm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*mm)
	
	kmx = tf.matrix_transpose(kxm)
	kmmi = tf.matrix_inverse(kmm+jitter)

	qmm_diag = tf.reduce_sum(tf.matmul(kxm,kmmi)*kxm,axis=2)

	gd = noise[:,tf.newaxis]
	gid = 1/gd
	
	tr = sfs*tf.cast(X,tf.float64)-tf.reduce_sum(qmm_diag,axis=1)
	
	kmx_gi_kxm = tf.matmul(kmx,kxm)/noise[:,tf.newaxis,tf.newaxis]

	giy = gid*y
	kgiy = tf.reduce_sum(kmx*giy[:,tf.newaxis,:],axis=2)

	inner = kmmi+kmx_gi_kxm+jitter

	covd = logdet(inner)+logdet(kmm+jitter)+tf.log(noise)*tf.cast(X,dtype=tf.float64)#tf.reduce_sum(tf.log(gd),axis=1)
	
	t1 = .5*tf.cast(X,tf.float64)*logtp
	t2 = .5*tf.reduce_sum(y*y*gid,axis=1) - .5*tf.reduce_sum(tf.reduce_sum(kgiy[:,:,tf.newaxis]*tf.matrix_inverse(inner),axis=1)*kgiy,axis=1)
	t3 = .5*covd
	t4 = .5*tf.div(tr,noise)

	return t1+t2+t3+t4

def lmult(diag,mat,naxes=3):
	return tf.expand_dims(diag,axis=naxes-1)*mat

def rmult(mat,diag,naxes=3):
	return mat*tf.expand_dims(diag,axis=naxes-2)

def gp_np(x,y,z):
	
	z = z.T
	ym = np.tile(np.expand_dims(np.expand_dims(y,0),2),(np.shape(z)[1],1,1))
	
	sls = np.abs(z[0])[:,np.newaxis,np.newaxis]
	sfs = np.abs(z[1])[:,np.newaxis,np.newaxis]
	#noise = (np.abs(z[2])+.1)[:,np.newaxis,np.newaxis]
	noise = .000001
		
	xx = pairwise_distance_np(x,x)[np.newaxis,:,:]
	kxx = sfs*np.exp(-.5*xx/sls)
	
	X = np.shape(xx)[-1]
	
	logtp = np.log(2.*np.pi)
	
	cov = kxx+noise*np.eye(X)[np.newaxis,:,:]
	top = np.squeeze(-.5*np.matmul(np.matmul(ym.transpose((0,2,1)),np.linalg.inv(cov)),ym))
	bot = .5*X*logtp+.5*np.linalg.slogdet(cov)[1]
	
	return bot-top

def gp_pred(x,y,t,z):

	z = tf.transpose(z)

	P = tf.shape(z)[1] # number of particles

	y = tf.tile(tf.expand_dims(y,0),(P,1))
	x = tf.tile(tf.expand_dims(x,0),(P,1,1))
	t = tf.tile(tf.expand_dims(t,0),(P,1,1))
	
	D = tf.shape(x)[2]
	X = tf.shape(x)[1]
	
	sls = z[0]
	sfs = z[1]
	noise = z[2]

	jitter = 1e-6*tf.eye(X,dtype=tf.float64)[tf.newaxis,:,:]

	xx = pairwise_distance(x,x)

	kxx = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*xx/sls[:,tf.newaxis,tf.newaxis])

	tx = pairwise_distance(t,x)

	ktx = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*tx/sls[:,tf.newaxis,tf.newaxis])

	kk = tf.matmul(ktx,tf.matrix_inverse(kxx+noise[:,tf.newaxis,tf.newaxis]+jitter))

	return tf.reduce_sum(kk*y[:,tf.newaxis,:],axis=2)

def gp(x,y,z):
	
	z = tf.transpose(z)
	x = tf.expand_dims(x,0)
	ym = tf.tile(tf.expand_dims(tf.expand_dims(y,0),2),(tf.shape(z)[1],1,1))
	
	sls = z[0][:,tf.newaxis,tf.newaxis]
	sfs = z[1][:,tf.newaxis,tf.newaxis]
	noise = z[2][:,tf.newaxis,tf.newaxis]
		
	xx = pairwise_distance(x,x)#[tf.newaxis,:,:]
	kxx = sfs*tf.exp(-.5*xx/sls)
	
	X = tf.shape(x)[1]
	
	logtp = tf.constant(np.log(2.*np.pi),dtype=tf.float64)
	
	cov = kxx+noise*tf.eye(X,dtype=tf.float64)#[tf.newaxis,:,:]

	jitter = 1e-6*tf.eye(X,dtype=tf.float64)[tf.newaxis,:,:]

	top = tf.squeeze(-.5*tf.matmul(tf.matmul(tf.matrix_transpose(ym),tf.matrix_inverse(cov+jitter)),ym))
	bot = .5*tf.cast(X,tf.float64)*logtp+.5*logdet(cov)
	
	return bot-top

def plot_pdf(U_z,npoints,*args,uselog=False):
	mesh_z1, mesh_z2, points = gridpoints(npoints,*args)
	z_pp  = tf.placeholder(tf.float64, [None, len(args)])
	if uselog:
		#out = U_z(z_pp)
		#prob = -out[0]
		#ge = out[1]
		prob = -U_z(z_pp)
	else:
		#out = U_z(z_pp)
		#prob = tf.exp(-out[0])
		#ge = out[1]
		prob = tf.exp(-U_z(z_pp))
	with tf.Session() as s:
		#phat_z, ge_val = s.run([prob, ge], feed_dict={z_pp: points} ) 
		phat_z = s.run(prob, feed_dict={z_pp: points} ) 
	print(phat_z)
	phat_z=phat_z.reshape([npoints,npoints])
	#phat_z=phat_z/np.abs(phat_z).max()
	plt.pcolormesh(mesh_z1, mesh_z2, phat_z)
	if uselog:
		z_min, z_max = np.nanmin(phat_z)-2*(np.nanmax(phat_z)-np.nanmin(phat_z)), np.nanmax(phat_z)
	else:
		z_min, z_max = -np.nanmax(phat_z), np.nanmax(phat_z)
	#print(phat_z,z_min,z_max)
	plt.pcolor(mesh_z1, mesh_z2, phat_z, cmap='RdBu', vmin=z_min, vmax=z_max)
	plt.xlim(args[0]); plt.ylim(args[1]); plt.title('Target distribution: $u(z)$')
	
	#return ge_val

def gridpoints(npoints,*args):
	sides = [np.linspace(arg[0],arg[1],npoints) for arg in args]
	coords = np.meshgrid(*sides)
	return coords[0], coords[1], np.hstack([coord.reshape(-1,1) for coord in coords])

def cluster_gp(n_clusters,pts_per_cluster,dim,var=None,lsqs=None,fsqs=None,noise=None):
	if not var:
		var = np.ones(n_clusters)*n_clusters/1e2
	if not fsqs:
		fsqs = np.random.permutation(1+np.arange(n_clusters))/1e2
	if not noise:
		noise = np.random.permutation(1+np.arange(n_clusters))/1e2
	if not lsqs:
		lsqs = (1+np.arange(n_clusters))**3/5e2
	
	domain = np.linspace(-n_clusters/2,n_clusters/2,n_clusters)
	centers = np.random.permutation(list(itertools.product(*[domain]*dim)))[:n_clusters]
	tx = [np.random.multivariate_normal(c,v*np.eye(dim),pts_per_cluster) for c,v in zip(centers,var)]
	
	pwd = [pairwise_distance_np(x,x) for x in tx]
	sqe = [fsq*np.exp(-.5*pw/v) for pw,v,fsq in zip(pwd,lsqs,fsqs)]
	eye = np.eye(pts_per_cluster)/1e6
	fs = [np.random.multivariate_normal(np.zeros(pts_per_cluster),sq+eye) for sq in sqe]
	ty = [f+ns*np.random.randn(*np.shape(f)) for f,ns in zip(fs,noise)]
	
	train_x = np.concatenate([x[:int(.8*len(x))] for x in tx])
	test_x = np.concatenate([x[int(.8*len(x)):] for x in tx])
	train_y = np.concatenate([y[:int(.8*len(y))] for y in ty])
	test_y = np.concatenate([y[int(.8*len(y)):] for y in ty])
	
	return train_x, train_y, test_x, test_y

from sgp_utils import pairwise_distance_np

def var(arr):
	return np.mean((arr - np.mean(arr))**2)

def l_init(arr,max_pairs=500):
	max_pairs = min(max_pairs,len(arr)//10)
	indices = np.random.permutation(np.arange(len(arr)))<max_pairs
	arr = arr[indices,:]
	pairs = pairwise_distance_np(arr,arr)
	pairs = pairs+1e8*np.eye(len(arr))
	#return np.median([np.sum((p[0]-p[1])**2) for p in pairs])
	return np.median(np.amin(pairs,axis=1))

def l_init_ard(arr,max_pairs=500):
	max_pairs = min(max_pairs,len(arr)//10)
	indices = np.random.permutation(np.arange(len(arr)))<max_pairs
	arr = arr[indices,:]
	ap = []
	for i in range(np.shape(arr)[1]):
		pairs = pairwise_distance_np(arr[:,i][:,np.newaxis],
			arr[:,i][:,np.newaxis])
		pairs = pairs+1e8*np.eye(len(arr))
		ap.append(np.median(np.amin(pairs,axis=1)))
	return np.array(ap)

def km_init(train_x,train_y,total_pips,n_regions,n_overlapping_pips=0,ard=False):

	ideal_region_size = len(train_x)//n_regions
	
	assert total_pips%n_regions == 0
	
	if n_regions>1:
		pips_per_region = total_pips//n_regions - n_overlapping_pips
	else:
		pips_per_region = total_pips

	min_region_size = int(np.sqrt(ideal_region_size*pips_per_region))
	
	D = np.shape(train_x)[1]
	
	# first, get regions
	rlabels, rcenters = km_clusters_sk(train_x,n_regions)
	
	lsqs = []
	noise = []
	pips = []
	fsqs = []

	widths = []
	
	for i in range(n_regions):
		region_size = np.sum(rlabels==i)
		region_points = train_x[rlabels==i]
		widths.append(np.sum((region_points-rcenters[i][np.newaxis,:])**2))
		#sumsqdist = tf.reduce_sum((m-centers[:,tf.newaxis,:])**2,axis=[1,2])
		if region_size<min_region_size:
			print('Pulling other points')
			candidates = train_x[rlabels!=i]
			topc = sorted(candidates,key=lambda x: np.sum((x-rcenters[i])**2))
			topc = topc[:min_region_size-region_size]
			region_points = np.vstack([region_points,topc])
		indices = np.random.permutation(np.arange(len(region_points)))<pips_per_region
		rpips = region_points[indices,:]
		#_, rpips = km_clusters_sk(train_x[rlabels==i],pips_per_region)
		pips.append(rpips.flatten())
		noise.append(var(train_y[rlabels==i])/100)
		fsqs.append(np.mean(train_y[rlabels==i]**2)+var(train_y[rlabels==i]))
		if ard:
			lsqs.append(l_init_ard(train_x[rlabels==i])/2)
		else:
			lsqs.append(l_init(train_x[rlabels==i])/2)
	if ard:
		lsqs = np.array(lsqs)
	else:
		lsqs = np.array(lsqs)[:,np.newaxis]
	fsqs = np.array(fsqs)[:,np.newaxis]
	noise = np.array(noise)[:,np.newaxis]
	pips = np.array(pips)
	
	if (n_regions>1) and (n_overlapping_pips>0):	
		for i in range(n_regions):
			other_pips = np.concatenate(pips[:i]+pips[i+1:])
			extra_pips = sorted(other_pips,key=lambda z: np.sum((z-rcenters[i])**2))[:n_overlapping_pips]
			pips[i] = np.concatenate([pips[i],np.array(extra_pips).flatten()])
		
	return np.hstack([lsqs,fsqs,noise,pips]), np.median(widths)

def nlog_normal(z):
	mn = tf.reduce_mean(z,axis=0)[tf.newaxis,:]
	return .5*tf.reduce_sum((z-mn)**2)

def km_clusters_sk(arr,n_clusters):
	from sklearn.cluster import MiniBatchKMeans
	kmr = MiniBatchKMeans(n_clusters=n_clusters)
	labels = kmr.fit_predict(arr)
	centers = kmr.cluster_centers_
	return labels,centers

def km_clusters_tf(arr,n_clusters,max_iter=50):
	from tensorflow.contrib.factorization import KMeans
	X = tf.placeholder(tf.float32, shape=[None, np.shape(arr)[1]])
	km = KMeans(inputs=X,num_clusters=n_clusters,
		distance_metric='squared_euclidean',use_mini_batch=True)
	training_graph = km.training_graph()
	if len(training_graph) > 6: # Tensorflow 1.4+
		(all_scores, cluster_idx, scores, cluster_centers_initialized,
			cluster_centers_var, init_op, train_op) = training_graph
	else:
		(all_scores, cluster_idx, scores, cluster_centers_initialized,
			init_op, train_op) = training_graph
	cluster_idx = cluster_idx[0]
	avg_distance = tf.reduce_mean(scores)
	init_vars = tf.global_variables_initializer()
	last_dist = np.mean(np.sum((arr - arr[0][np.newaxis,:])**2,axis=1))+1
	d = last_dist-1
	i=0
	with tf.Session() as s:
		s.run(init_vars, feed_dict={X: arr})
		initialized = False
		while initialized==False:
			_, initialized = s.run([init_op,cluster_centers_initialized], feed_dict={X: arr})
		while ((d<last_dist) and (i<max_iter)):
			last_dist = d
			_, d, idx = s.run([train_op, avg_distance, cluster_idx],
				feed_dict={X: arr})
			i = i+1
	cluster_centers = [np.mean(arr[idx==i,:],axis=0) for i in range(n_clusters)]
	return idx, np.array(cluster_centers)