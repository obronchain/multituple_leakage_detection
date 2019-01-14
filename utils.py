import numpy as np
usage ="""
    options:
        fast  - reduced number of samples (no averaging)
        paper - plots form the paper
        file  - uses .pkl files and only generate plots"""


def get_HW(tabular):
	""" return the HW of 8bits values array """
	HW_val = np.array([bin(n).count("1") for n in range(0,256)],dtype=int)
	return HW_val[tabular]

def my_cov(nl=32,sigma=16,slope=-1/32,init=1):
	""" Generate covariance matrix """
	cov = np.identity(nl)
	for i in range(0,nl):
		for j in range(i+1,nl):
			cov[i,j] = np.max([init+np.abs(i-j)*slope,0])
			cov[j,i] = cov[i,j]
	return cov*(sigma**2)

def set_density(Vx,Vy,density=1):
    """ Set density to given inputs vectors """
    nl = len(Vx)
    if density==1:
        return Vx,Vy
    no = int(nl*(1-density))
    Vx_ret = np.array(Vx)
    Vy_ret = np.array(Vy)
    Vx_ret[-no:] = Vy_ret[-no:]

    return Vx_ret,Vy_ret
