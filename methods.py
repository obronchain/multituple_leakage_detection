import numpy as np
import scipy.stats

####################################### Detection methods
def p_value_Ttest(t,N=1500,df=1000,CDF=scipy.stats.t.cdf):
	alphaTH = 2*(CDF(np.abs(t),df)-1)
	alpha = ((alphaTH+1)**N)
	return 1-alpha

def Ttest(trace1,trace2,alpha=1E-5,getp=True):
	""" computes a Welch t test for each of the time sample
	between the two traces """
	nl = len(trace1[0,:])
	m1 = np.mean(trace1,axis=0)
	m2 = np.mean(trace2,axis=0)
	var1 = np.var(trace1,axis=0)
	var2 = np.var(trace2,axis=0)

	Nx = len(trace1[:,0])
	Ny = len(trace2[:,0])

	t = (m1-m2)/np.sqrt(var1/Nx + var2/Ny)
	if getp:
		v = ((var1/Nx + var2/Ny)**2) /(((var1/Nx)**2)/(Nx-1) + ((var2/Ny)**2)/(Ny-1))
		p = p_value_Ttest(t,N=nl,df=np.mean(v))
		alphaTH = 1-(1-alpha)**(1/nl)
		TH = np.abs(scipy.stats.t.ppf((1-alphaTH/2),np.mean(v)))
	else:
		p = None
		alphaTH = 1-(1-alpha)**(1/nl)
		TH = np.abs(scipy.stats.t.ppf((1-alphaTH/2),10000))

	alphaTH = 1-(1-alpha)**(1/nl)
	return t,p,TH

def Tsquare(t1,t2,alpha=1E-5,getp=True):
	X = t1.T
	Y = t2.T
	nx = len(X[0,:])
	ny = len(Y[0,:])
	p = len(X[:,0])

	D  = np.broadcast_to(np.mean(X,axis=1) - np.mean(Y,axis=1), (1,p)).T
	Sx = np.cov(X)
	Sy = np.cov(Y)
	S = (((nx-1)*Sx) + ((ny-1)*Sy))/(nx+ny-2)
	tsquare = np.dot(D.T ,np.dot(np.linalg.inv(S),D))*(nx*ny/(nx+ny))
	F = (((nx+ny-p-1)/(p*(nx+ny-2))))*tsquare
	th = scipy.stats.f.ppf(1-alpha,p,nx+ny-1-p)

	if getp:
		p_value = 1-scipy.stats.f.cdf(F,p,nx+ny-1-p)
	else:
		p_value = False

	return F,p_value,th

def Tsquare_minp(t1,t2,alpha=1E-5,getp=True,N_split = 10):
	nl = len(t1[0,:])
	p = int(nl/N_split)
	nx = len(t1[:,0])
	ny = len(t2[:,0])
	indexes_limit = np.linspace(0,nl,N_split+1,dtype=int)

	F_s = np.zeros(N_split)

	for i in range(N_split):
		indexes = range(indexes_limit[i],indexes_limit[i+1])
		F_s[i],_,_ = Tsquare(t1[:,indexes],t2[:,indexes],getp=False)

	i = np.argmax(F_s)
	F = F_s[i]
	alphaTH = 1-(1-alpha)**(1/N_split)
	TH = scipy.stats.f.ppf((1-alphaTH),p,nx+ny-1-p)

	CDF = scipy.stats.f.cdf
	alphaTH = CDF(F,p,nx+ny-1-p)
	p = 1-((alphaTH)**N_split)
	return F,p,TH
	
def MD(t1,t2,sigma=None,alpha=1E-5,getp=True):
	sigma1 = np.std(t1,axis=0)
	sigma2 = np.std(t2,axis=0)

	Nx = len(t1[:,0])
	Ny = len(t2[:,0])
	std = np.sqrt((sigma2**2)/Ny+(sigma1**2)/Nx)
	X = np.mean(t1,axis=0)-np.mean(t2,axis=0)
	t = np.sum(np.power(X/std,2))

	df = len(t1[0,:])
	if getp:
		p = 1-scipy.stats.chi2.cdf(t,df)
	else:
		p = None
	return t,p,scipy.stats.chi2.ppf(1-alpha,df)
