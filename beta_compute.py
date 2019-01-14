import numpy as np
import scipy.stats
import scipy.special
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
from methods import *
from utils import *


def beta_Hotteling(diffs,Nx,Ny,Sigma,alpha=1E-5,identity=False):
	"""" Compute beta for Hotelling test """
	nl = len(diffs)
	if identity == True:
		effect_size = ((Nx * Ny) / (Nx + Ny)) * np.dot(diffs.T,diffs)/Sigma
	else:
		effect_size = ((Nx * Ny) / (Nx + Ny)) * np.dot(diffs.T,np.dot(np.linalg.inv(Sigma),diffs))
	th_f = scipy.stats.f.ppf(1-alpha,nl,Nx+Ny-2)
	beta = scipy.special.ncfdtr(nl,Nx+Ny-2,effect_size,th_f)
	return beta

def beta_TVLA(diffs,Nx,Ny,sigma2,alpha=1E-5,v=10000,sigma2b=None):
	"""" Compute beta for TVLA method """
	if sigma2b is None:
		sigma2b = sigma2
	nl = len(diffs)
	alphaTH = 1-(1-alpha)**(1/nl)
	TH_indep = np.abs(scipy.stats.t.ppf((1-alphaTH/2),v))
	beta = 1
	d = diffs
	effect_size = np.abs(d)/np.sqrt((sigma2 /Nx + sigma2 / Ny))
	df = v
	power = stats.nct.sf(TH_indep,df,effect_size) + stats.nct.cdf(-TH_indep,df,effect_size)
	beta = np.product(1- power)
	return beta

def SNRvsN(SNRs,nl,betas=[1E-3,1E-4],Nav=1,alpha=1E-5):
	""" get the number of traces at a given SNR """
	sigma2s = 2/SNRs
	Ns = np.zeros((len(betas),2,len(sigma2s)))

	for av in tqdm(range(Nav)):
		Vx = get_HW(np.random.randint(0,256,nl,dtype=np.uint8))
		Vy = get_HW(np.random.randint(0,256,nl,dtype=np.uint8))

		for i,beta in enumerate(betas):
			starting_points = [10,10]
			end_points = [10,10]

			for s,sigma2 in enumerate(sigma2s):
				######Hotelling's test
				def func(N,sigma=sigma2,nl=nl,beta=beta,diffs=Vx-Vy,alpha=alpha):
					return beta - beta_Hotteling(diffs,N,N,np.identity(nl)*sigma2,alpha)
				#ensure correct internal
				while func(starting_points[0]) >0:
					starting_points[0] *= 0.8
				while func(end_points[0]) < 0:
					end_points[0] *= 1.2
				Ns[i,0,s] += scipy.optimize.brenth(func,starting_points[0],end_points[0])

				#####T-test
				def func(N,sigma=sigma2,nl=nl,beta=beta,diffs=Vx-Vy,alpha=alpha):
					return beta - beta_TVLA(diffs,N,N,sigma2,alpha)
				#ensure correct internal
				while func(starting_points[1]) >0:
					starting_points[1] *= 0.8
				while func(end_points[1]) < 0:
					end_points[1] *= 1.2

				Ns[i,1,s] += scipy.optimize.brenth(func,starting_points[1],end_points[1])
				starting_points[1] = Ns[i,1,s]

	return Ns/Nav

def NlvsN(nls,SNR,betas=[1E-3,1E-4],Nav=1,alpha=1E-5):
	""" get the number of traces for a given trace length """
	sigma2 = 2/SNR
	Ns = np.zeros((len(betas),2,len(nls)))
	nls_max = np.max(nls)

	for av in tqdm(range(Nav)):
		Vx = get_HW(np.random.randint(0,256,nls_max,dtype=np.uint8))
		Vy = get_HW(np.random.randint(0,256,nls_max,dtype=np.uint8))
		diff = Vx-Vy
		while diff[0] == 0:
			Vx = get_HW(np.random.randint(0,256,nls_max,dtype=np.uint8))
			diff = Vx - Vy
		for i,beta in enumerate(betas):
			starting_points = [10,10]
			end_points = [10,10]

			for s,nl in enumerate(tqdm(nls)):
				######Hotelling's test
				def func(N,sigma=sigma2,nl=nl,beta=beta,diffs=Vx-Vy,alpha=alpha):
					return beta - beta_Hotteling(diffs[:nl],N,N,sigma2,alpha,identity=True)

				#ensure correct interval
				it = 0

				while func(starting_points[0]) >0:
					starting_points[0] *= 0.8
					# print("it %d"%(it))
					# it+=1
				it = 0
				while func(end_points[0]) < 0:
					end_points[0] *= 1.2
					# print("it %d"%(it))
					# it+=1
				Ns[i,0,s] += scipy.optimize.brenth(func,starting_points[0],end_points[0])

				######T-test
				def func(N,sigma=sigma2,nl=nl,beta=beta,diffs=Vx-Vy,alpha=alpha):
					return beta - beta_TVLA(diffs[:nl],N,N,sigma2,alpha)

				#ensure correct interval
				it = 0
				while func(starting_points[1]) >0:
					starting_points[1] *= 0.8
					# print("it %d"%(it))
					# it+=1
				while func(end_points[1]) < 0:
					end_points[1] *= 1.2
					# print("it %d"%(it))
					# it+=1

				Ns[i,1,s] += scipy.optimize.brenth(func,starting_points[1],end_points[1])

	return Ns/Nav

def DensityvsN(densities,nl,SNR,betas=[1E-3,1E-4],Nav=1,alpha=1E-5):
	""" get the number of traces for a given trace length """
	sigma2 = 2/SNR
	Ns = np.zeros((len(betas),2,len(densities)))

	for av in tqdm(range(Nav)):
		Vx = get_HW(np.random.randint(0,256,nl,dtype=np.uint8))
		Vy = get_HW(np.random.randint(0,256,nl,dtype=np.uint8))
		diff = Vx-Vy
		while diff[0] == 0:
			Vx = get_HW(np.random.randint(0,256,nl,dtype=np.uint8))
			diff = Vx - Vy
		for i,beta in enumerate(betas):
			starting_points = [10,10]
			end_points = [10,10]

			for s,d in enumerate(densities):
				######Hotelling's test
				Vx_d,Vy_d = set_density(Vx,Vy,density=d)
				diffs = Vx_d - Vy_d
				def func(N,sigma=sigma2,beta=beta,diffs=diffs,alpha=alpha):
					return beta - beta_Hotteling(diffs,N,N,np.identity(nl)*sigma2,alpha)

				#ensure correct interval
				while func(starting_points[0]) >0:
					starting_points[0] *= 0.8
				while func(end_points[0]) < 0:
					end_points[0] *= 1.2
				Ns[i,0,s] += scipy.optimize.brenth(func,starting_points[0],end_points[0])

				######T-test
				def func(N,sigma=sigma2,beta=beta,diffs=diffs,alpha=alpha):
					return beta - beta_TVLA(diffs,N,N,sigma2,alpha)

				#ensure correct interval
				while func(starting_points[1]) >0:
					starting_points[1] *= 0.8
				while func(end_points[1]) < 0:
					end_points[1] *= 1.2

				Ns[i,1,s] += scipy.optimize.brenth(func,starting_points[1],end_points[1])

	return Ns/Nav
