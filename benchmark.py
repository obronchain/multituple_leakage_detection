from methods import *
import numpy as np
import time
from tqdm import tqdm
from utils import *
import matplotlib.pyplot as plt
from settings import *

def timeVsnl(nls=np.linspace(50,200,50,dtype=int),Nx=200,Ny=200,Nav=100,getp=False):
    """ Computes CPU with increasing nl"""
    times = np.zeros((3,len(nls)))

    for nav in tqdm(range(Nav)):
        for i,nl in enumerate(nls):
            X = get_HW(np.random.randint(0,256,nl)) + np.random.normal(0,40,size=(Nx,nl))
            Y = get_HW(np.random.randint(0,256,nl)) + np.random.normal(0,40,size=(Nx,nl))

            #Ttest
            start = time.clock()
            t,_,_ = Ttest(X,Y,getp=getp)
            end = time.clock()
            times[0,i] += end-start

            #Hotelling
            start = time.clock()
            t,_,_ = Tsquare(X,Y,getp=getp)
            end = time.clock()
            times[1,i] += end-start

            #Dtest
            start = time.clock()
            t,_,_ = MD(X,Y,getp=getp)
            end = time.clock()
            times[2,i] += end-start

    times = times/Nav
    return times

def timeVsN(Ns=np.linspace(50,500,50,dtype=int),Nav=100,getp=False,nl=200):
    """ Computes CPU with increasing N"""
    times = np.zeros((3,len(Ns)))
    for nav in tqdm(range(Nav)):
        for i,N in enumerate(Ns):
            X = get_HW(np.random.randint(0,256,nl)) + np.random.normal(0,40,size=(N,nl))
            Y = get_HW(np.random.randint(0,256,nl)) + np.random.normal(0,40,size=(N,nl))

            #Ttest
            start = time.clock()
            t,_,_ = Ttest(X,Y,getp=getp)
            end = time.clock()
            times[0,i] += end-start

            #Hotelling
            start = time.clock()
            t,_,_ = Tsquare(X,Y,getp=getp)
            end = time.clock()
            times[1,i] += end-start

            #Dtest
            start = time.clock()
            t,_,_ = MD(X,Y,getp=getp)
            end = time.clock()
            times[2,i] += end-start

    times = times/Nav
    return times
