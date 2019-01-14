import numpy as np
import scipy.stats
import scipy.special
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
from methods import *
from utils import *

def sample_Tests(Vx,Vy,Sigma,nl,Nx,Ny,
			getTtest=True,getDtest=True,getHot=True,Sigmab=None,getHotminp=False,N_split=1):
	""" Sample the test statistics for a given Vx and Vy"""
	if Sigmab is None:
		Sigmab = Sigma

	statistic = [0,0,0,0]
	X = Vx + np.random.multivariate_normal(np.zeros(nl),Sigma,Nx)
	Y = Vy + np.random.multivariate_normal(np.zeros(nl),Sigmab,Ny)

	#T-test
	if getTtest:
		t,p,th = Ttest(X,Y)
		statistic[0] = max(t,key=abs)

	#Hotelling
	if getHot:
		t2,p,th = Tsquare(X,Y)
		statistic[1] = t2

	#D-test
	if getDtest:
		t2,p,th = MD(X,Y)
		statistic[2] = t2

	#Hotelling
	if getHotminp:
		t2,p,th = Tsquare_minp(X,Y,N_split=N_split)
		statistic[3] = t2

	return statistic

def sample_distribution(Sigma,nl,Nx,Ny,NSamples=1000, hypothesis='null',
				getTtest=True,getDtest=True,getHot=True,Sigmab=None,density=1,random=False,getHotminp=False,N_split=1):
	""" sample the test statistic distribution. Random only works for density = 1"""

	samples = np.zeros((4,NSamples))

	for ns in tqdm(range(NSamples)):
		Vx = get_HW(np.random.randint(0,256,nl))
		if hypothesis == 'null':
			Vy = Vx
		else:
			Vy = get_HW(np.random.randint(0,256,nl))

		if random:
			Vy = get_HW(np.random.randint(0,256,(Ny,nl)))
		else:
			Vx,Vy = set_density(Vx,Vy,density=density)

		samples[:,ns] = sample_Tests(Vx,Vy,Sigma,nl,Nx,Ny,Sigmab=Sigmab,
								getTtest=getTtest,getDtest=getDtest,getHot=getHot,
								getHotminp=getHotminp,N_split=N_split)

	return samples

def compare_TVLA_original(Sigma,nl,Nx,NSamples,TH=4.5,TH_cor=4.5):

	distri = sample_distribution(Sigma,nl,Nx,Nx,NSamples,getDtest=False,getHot=False)
	betas_orig = np.sum(np.abs(distri[0,:])>TH)/NSamples

	distri_1 = sample_distribution(Sigma,nl,int(Nx/2),int(Nx/2),NSamples,getDtest=False,getHot=False)
	distri_2 = sample_distribution(Sigma,nl,int(Nx/2),int(Nx/2),NSamples,getDtest=False,getHot=False)

	betas_TVLA = np.sum((np.abs(distri_1[0,:])>TH_cor) & (np.abs(distri_2[0,:])>TH_cor))/NSamples
	betas_TVLA_uncorrect = np.sum((np.abs(distri_1[0,:])>TH) & (np.abs(distri_2[0,:])>TH))/NSamples

	betas_cor = np.sum((np.abs(distri_1[0,:])>TH_cor))/NSamples
	return betas_orig,betas_TVLA,betas_cor,betas_TVLA_uncorrect

def compare_fix_vs_random(Sigma,nl,Nx,NSamples,alpha=1E-5):
	N = np.linspace(100,Nx,dtype=int)
	nl = len(Sigma[0,:])
	alphaTH = 1-(1-alpha)**(1/nl)
	TH_indep_T = np.abs(scipy.stats.t.ppf((1-alphaTH/2),10000))
	TH_indep_D = scipy.stats.chi2.ppf(1-alpha,nl)

	betas_r = np.zeros((2,len(N)))
	betas_f = np.zeros((2,len(N)))
	betas_f_2 = np.zeros((2,len(N)))
	betas_f_3 = np.zeros((2,len(N)))

	getDtest = True
	getTtest = True

	for i,n in enumerate(tqdm(N)):
		distri = sample_distribution(Sigma,len(Sigma[0,:]),n,n,NSamples=NSamples,hypothesis='alt',random=True,getDtest=getDtest,getTtest=getTtest)
		betas_r[0,i] = np.sum(np.abs(distri[0,:])<TH_indep_T)
		betas_r[1,i] = np.sum(np.abs(distri[2,:])<TH_indep_D)

	for i,n in enumerate(tqdm(N)):
		distri = sample_distribution(Sigma,len(Sigma[0,:]),n,n,NSamples=NSamples,hypothesis='alt',random=False,getDtest=getDtest,getTtest=getTtest)
		betas_f[0,i] = np.sum(np.abs(distri[0,:])<TH_indep_T)
		betas_f[1,i] = np.sum(np.abs(distri[2,:])<TH_indep_D)


	alpha = np.sqrt(alpha)
	alphaTH = 1-(1-alpha)**(1/nl)
	TH_indep_T_cor = np.abs(scipy.stats.t.ppf((1-alphaTH/2),10000))
	TH_indep_D_cor = scipy.stats.chi2.ppf(1-alpha,nl)

	for i,n in enumerate(tqdm(N)):
		distri_1 = sample_distribution(Sigma,len(Sigma[0,:]),int(n/2),int(n/2),NSamples=NSamples,hypothesis='alt',random=False,getDtest=getDtest,getTtest=getTtest)
		distri_2 = sample_distribution(Sigma,len(Sigma[0,:]),int(n/2),int(n/2),NSamples=NSamples,hypothesis='alt',random=False,getDtest=getDtest,getTtest=getTtest)
		betas_f_2[0,i] = np.sum((np.abs(distri_1[0,:])>TH_indep_T_cor) & (np.abs(distri_2[0,:])>TH_indep_T_cor))
		betas_f_2[1,i] = np.sum((np.abs(distri_2[2,:])>TH_indep_D_cor) & (np.abs(distri_2[2,:])>TH_indep_D_cor))

		betas_f_3[0,i] = np.sum((np.abs(distri_1[0,:])>TH_indep_T) & (np.abs(distri_2[0,:])>TH_indep_T))
		betas_f_3[1,i] = np.sum((np.abs(distri_2[2,:])>TH_indep_D) & (np.abs(distri_2[2,:])>TH_indep_D))


	return betas_r/NSamples,betas_f/NSamples,1-(betas_f_2/NSamples),1-(betas_f_3/NSamples),N


def loss(Sigma,nl,Nx,Ny,NSamples=10000,alpha=1E-1,N_eval=1000,getTtest=False,getDtest=False):
	""" Compute the loss of the Ttest due to dependent signal"""
	if getTtest:
		alphaTH = 1-(1-alpha)**(1/nl)
		TH_indep = np.abs(scipy.stats.t.ppf((1-alphaTH/2),10000))
	else:
		TH_indep = scipy.stats.chi2.ppf(1-alpha,nl)

	if getTtest:
		index = 0
	elif getDtest:
		index = 2
	else:
		print("No flag correct")
		return

	### getting sampled threshold
	print("Sampling threshold")
	distri_null = sample_distribution(Sigma,nl,Nx,Ny,NSamples,
				hypothesis='null',getHot=False,getDtest=getDtest,getTtest=getTtest)
	distri_null = np.sort(np.abs(distri_null[index,:]))
	TH_sampled = np.abs(distri_null[int(NSamples*(1-alpha))])
	Ns = np.linspace(100,Nx,20,dtype=int)
	betas = np.zeros((2,len(Ns)))

	print("Sampling loss")
	### Computing
	for i,ns in enumerate(Ns):
		distri_null = sample_distribution(Sigma,nl,ns,ns,N_eval,
				hypothesis='alternative',getHot=False,getDtest=getDtest,getTtest=getTtest)
		betas[0,i] = np.sum(np.abs(distri_null[index,:])<TH_indep)
		betas[1,i] = np.sum(np.abs(distri_null[index,:])<TH_sampled)

	return betas/N_eval,Ns
