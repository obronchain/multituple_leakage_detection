#! /bin/python3
from settings import *
from beta_compute import *
import pickle
from sampled_experiments import *
from benchmark import *
import sys
from utils import *

def fig5b(nl=64,betas=[1E-3,1E-4],Nav=50,SNRs=np.logspace(-1,-3,20),file=None):
    print("------------Figure 5b-----------")
    #loading file
    if file is None:
        Ns = SNRvsN(SNRs,nl,betas=betas,Nav=Nav)
    else:
        dic = pickle.load(open(file,"rb"))
        Ns = dic["Ns"]
        betas = dic["betas"]
        nl = dic["nl"]
        Nav = dic["Nav"]
        SNRs = dic["SNRs"]

    plt.figure()
    plt.loglog(SNRs[:30],2*Ns[1,0,:],
            label=r'$\beta = 10^{-4}$, '+labels["Dtest"],color=colors["Dtest"])
    plt.loglog(SNRs[:30],2*Ns[1,1,:],
            label=r'$\beta = 10^{-4}$, '+labels["Ttest"],color=colors["Ttest"])
    plt.loglog(SNRs[:30],2*Ns[0,0,:],'--',
            label=r'$\beta = 10^{-3}$, '+labels["Dtest"],color=colors["Dtest"])
    plt.loglog(SNRs[:30],2*Ns[0,1,:],'--',
            label=r'$\beta = 10^{-3}$, '+labels["Ttest"],color=colors["Ttest"])

    plt.xlabel('SNR')
    plt.ylabel(r'$N$')
    plt.grid(True,which="both",ls="-")
    plt.legend()
    plt.savefig("fig5b.pdf",bbox_inches = 'tight', pad_inches = 0)

    dic = {"Ns":Ns,
    'betas':betas,
    'nl':nl,
    'Nav':Nav,
    'SNRs':SNRs}
    pickle.dump(dic,open("fig5b.pkl","wb"))


def fig5a(nls=np.logspace(0,6,10,dtype=int),betas=[1E-3,1E-4],Nav=50,SNR=0.01,file=None):
    print("------------Figure 5a-----------")

    #loading file
    if file is None:
        Ns = NlvsN(nls,SNR=SNR,Nav=Nav)
    else:
        dic = pickle.load(open(file,"rb"))
        Ns = dic["Ns"]
        betas = dic["betas"]
        nls = dic["nls"]
        Nav = dic["Nav"]
        SNR = dic["SNR"]

    plt.figure()
    plt.loglog(nls,2*Ns[1,0,:],
                label=r'$\beta = 10^{-4}$, '+labels["Dtest"],color=colors["Dtest"])
    plt.loglog(nls,2*Ns[1,1,:],
                label=r'$\beta = 10^{-4}$, '+labels["Ttest"],color=colors["Ttest"])
    plt.loglog(nls,2*Ns[0,0,:],'--',
                label=r'$\beta = 10^{-3}$, '+labels["Dtest"],color=colors["Dtest"])
    plt.loglog(nls,2*Ns[0,1,:],'--',
                label=r'$\beta = 10^{-3}$, '+labels["Ttest"],color=colors["Ttest"])

    plt.xlabel(r'\huge$n_l$')
    plt.ylabel(r'$N$')
    plt.grid(True,which="both",ls="-")
    plt.legend()

    plt.savefig("fig5a.pdf",bbox_inches = 'tight', pad_inches = 0)
    dic = {"Ns":Ns,
    'betas':betas,
    'nls':nls,
    'Nav':Nav,
    'SNR':SNR}

    pickle.dump(dic,open("fig5a.pkl","wb"))

def fig6(densities=np.logspace(np.log10(1/256),np.log10(1),10),
                        nl=256,betas=[1E-3,1E-4],Nav=200,SNR=0.01,file=None):
    print("------------Figure 6-----------")
    #loading file
    if file is None:
        Ns = DensityvsN(densities,nl,SNR=SNR,Nav=Nav)
    else:
        dic = pickle.load(open(file,"rb"))
        Ns = dic["Ns"]
        betas = dic["betas"]
        nl = dic["nl"]
        Nav = dic["Nav"]
        SNR = dic["SNR"]

    plt.figure()
    plt.loglog(densities,2*Ns[1,0,:],
                label=r'$\beta = 10^{-4}$, '+labels["Dtest"],color=colors["Dtest"])
    plt.loglog(densities,2*Ns[1,1,:],
                label=r'$\beta = 10^{-4}$, '+labels["Ttest"],color=colors["Ttest"])
    plt.loglog(densities,2*Ns[0,0,:],'--',
                label=r'$\beta = 10^{-3}$, '+labels["Dtest"],color=colors["Dtest"])
    plt.loglog(densities,2*Ns[0,1,:],'--',
                label=r'$\beta = 10^{-3}$, '+labels["Ttest"],color=colors["Ttest"])

    plt.xlabel(r'\huge$\phi$')
    plt.ylabel(r'$N$')
    plt.grid(True,which="both",ls="-")
    plt.legend()
    plt.savefig("fig6.pdf",bbox_inches = 'tight', pad_inches = 0)

    dic = {"Ns":Ns,
    'betas':betas,
    'nl':nl,
    'densities':densities,
    'Nav':Nav,
    'SNR':SNR}

    pickle.dump(dic,open("fig6.pkl","wb"))
    return Ns,densities

def fig8(nl=32,Nx=2000,Ny=2000,NSamples=100000,SNR=0.01,alpha=0.1,file=None):
    print("------------Figure 7/8 Tab 1 -----------")

    if file is None:
        sigma = np.sqrt(2/SNR)
        COV1 = np.identity(nl)*(2/SNR)
        COV2 = my_cov(nl,sigma=sigma,slope=-0.1,init=1)
        COV3 = my_cov(nl,sigma=sigma,slope=-0.02,init=1)

        print("Sampling COV1")
        distri_1 = sample_distribution(COV1,nl,Nx,Ny,NSamples,hypothesis="null")
        print("Sampling COV2")
        distri_2 = sample_distribution(COV2,nl,Nx,Ny,NSamples,hypothesis="null")
        print("Sampling COV3")
        distri_3 = sample_distribution(COV3,nl,Nx,Ny,NSamples,hypothesis="null")
    else:
        dic = pickle.load(open(file,"rb"))
        SNR = dic["SNR"]
        distri_1 = dic["distri_1"]
        distri_2 = dic["distri_2"]
        distri_3 = dic["distri_3"]
        NSamples = len(distri_1[0,:])
        nl = dic["nl"]
        Nx = dic["Nx"]
        sigma = np.sqrt(2/SNR)
        COV1 = np.identity(nl)*(2/SNR)
        COV2 = my_cov(nl,sigma=sigma,slope=-0.1,init=1)
        COV3 = my_cov(nl,sigma=sigma,slope=-0.02,init=1)


    sigma = np.sqrt(2/SNR)
    plt.figure()
    plt.matshow(COV1)
    plt.colorbar()
    plt.savefig('fig7a.pdf',bbox_inches = 'tight', pad_inches = 0)
    plt.figure()
    plt.matshow(COV2)
    plt.colorbar()
    plt.savefig('fig7b.pdf',bbox_inches = 'tight', pad_inches = 0)
    plt.figure()
    plt.matshow(COV3)
    plt.colorbar()
    plt.savefig('fig7c.pdf',bbox_inches = 'tight', pad_inches = 0)


    # Estimated alpha
    alphaTH = 1-(1-alpha)**(1/nl)
    TH_Ttest = np.abs(scipy.stats.t.ppf((1-alphaTH/2),1/((1/Nx)+(1/Ny))))
    TH_T2 = scipy.stats.f.ppf(1-alpha,nl,Nx+Ny-1-nl)
    TH_Dtest = scipy.stats.chi2.ppf(1-alpha,nl)

    print("Estimated alpha (%0.3f): COV1 -  COV2 -  COV3"%(alpha))
    print("Ttest :   %0.3f  -    %0.3f   - %0.3f "%(np.sum(np.abs(distri_1[0,:])>TH_Ttest)/NSamples,
                                            np.sum(np.abs(distri_2[0,:])>TH_Ttest)/NSamples,
                                            np.sum(np.abs(distri_3[0,:])>TH_Ttest)/NSamples))
    print("Dtest :   %0.3f  -    %0.3f   - %0.3f "%(np.sum(distri_1[2,:]>TH_Dtest)/NSamples,
                                            np.sum(distri_2[2,:]>TH_Dtest)/NSamples,
                                            np.sum(distri_3[2,:]>TH_Dtest)/NSamples))
    print("T2test:   %0.3f  -    %0.3f   - %0.3f "%(np.sum(distri_1[1,:]>TH_T2)/NSamples,
                                            np.sum(distri_2[1,:]>TH_T2)/NSamples,
                                            np.sum(distri_3[1,:]>TH_T2)/NSamples))
    ############# PLOTS
    alpha=0.3

    # T-tEST
    plt.figure()
    plt.grid(True,which="both",ls="-")
    plt.hist(distri_1[0,:],bins=100,
            label=labels["Cov1"],density=True,alpha=alpha,color=colors["Cov1"])
    plt.hist(distri_2[0,:],bins=100,
            label=labels["Cov2"],density=True,color=colors["Cov2"],alpha=alpha)
    plt.hist(distri_3[0,:],bins=100,
            label=labels["Cov3"],density=True,color=colors["Cov3"],alpha=alpha)
    plt.axvline(TH_Ttest,color='r',linestyle='--')
    plt.axvline(-TH_Ttest,color='r',linestyle='--')
    plt.xlim((-5,5))
    plt.legend()
    plt.savefig('fig8-Ttest.pdf',bbox_inches = 'tight', pad_inches = 0)

    # D-test
    plt.figure()
    plt.grid(True,which="both",ls="-")
    plt.hist(distri_1[2,:],bins=np.linspace(0,162,100),
                label=labels["Cov1"],density=True,alpha=alpha,color=colors["Cov1"])
    plt.hist(distri_2[2,:],bins=np.linspace(0,162,100),
                label=labels["Cov2"],density=True,color=colors["Cov2"],alpha=alpha)
    plt.hist(distri_3[2,:],bins=np.linspace(0,162,100),
                label=labels["Cov3"],density=True,color=colors["Cov3"],alpha=alpha)
    plt.legend()
    plt.axvline(TH_Dtest,color='r',linestyle='--')
    plt.savefig('fig8-Dtest.pdf',bbox_inches = 'tight', pad_inches = 0)

    # Hotelling's T2-test
    plt.figure()
    plt.grid(True,which="both",ls="-")
    plt.hist(distri_1[1,:],bins=100,
            label=labels["Cov1"],density=True,alpha=alpha,color=colors["Cov1"])
    plt.hist(distri_2[1,:],bins=100,
            label=labels["Cov2"],density=True,color=colors["Cov2"],alpha=alpha)
    plt.hist(distri_3[1,:],bins=100,
            label=labels["Cov3"],density=True,color=colors["Cov3"],alpha=alpha)
    plt.legend()
    plt.axvline(TH_T2,color='r',linestyle='--')
    plt.savefig('fig8-Tsquare.pdf',bbox_inches = 'tight', pad_inches = 0)

    dic = {"SNR":SNR,
            'distri_1':distri_1,
            'distri_2':distri_2,
            'distri_3':distri_3,
            'nl':nl,
            'alpha':alpha,
            'Nx':Nx}
    pickle.dump(dic,open("fig8.pkl","wb"))


def tab_1(nl=320,Nx=2000,Ny=2000,NSamples=50000,SNR=0.01,alpha=0.1,file=None,N_split=10):
    print("------------Figure 8, Hot + min_p table -----------")

    if file is None:
        sigma = np.sqrt(2/SNR)
        COV1 = np.identity(nl)*(2/SNR)
        COV2 = my_cov(nl,sigma=sigma,slope=-0.04,init=1)
        COV3 = my_cov(nl,sigma=sigma,slope=-0.002,init=1)

        print("Sampling COV1")
        distri_1 = sample_distribution(COV1,nl,Nx,Ny,NSamples,hypothesis="null",getTtest=False,getHot=False,getDtest=False,getHotminp=True,N_split=N_split)
        print("Sampling COV2")
        distri_2 = sample_distribution(COV2,nl,Nx,Ny,NSamples,hypothesis="null",getTtest=False,getHot=False,getDtest=False,getHotminp=True,N_split=N_split)
        print("Sampling COV3")
        distri_3 = sample_distribution(COV3,nl,Nx,Ny,NSamples,hypothesis="null",getTtest=False,getHot=False,getDtest=False,getHotminp=True,N_split=N_split)
    else:
        dic = pickle.load(open(file,"rb"))
        SNR = dic["SNR"]
        distri_1 = dic["distri_1"]
        distri_2 = dic["distri_2"]
        distri_3 = dic["distri_3"]
        NSamples = len(distri_1[0,:])
        nl = dic["nl"]
        Nx = dic["Nx"]
        sigma = np.sqrt(2/SNR)
        COV1 = np.identity(nl)*(2/SNR)
        COV2 = my_cov(nl,sigma=sigma,slope=-0.04,init=1)
        COV3 = my_cov(nl,sigma=sigma,slope=-0.002,init=1)

    # Estimated alpha
    alphaTH = 1-(1-alpha)**(1/N_split)
    TH_T2_minp = scipy.stats.f.ppf((1-alphaTH),int(nl/N_split),Nx+Ny-1-int(nl/N_split))

    print("T2test_minp:   %0.3f  -    %0.3f   - %0.3f "%(np.sum(distri_1[3,:]>TH_T2_minp)/NSamples,
                                            np.sum(distri_2[3,:]>TH_T2_minp)/NSamples,
                                            np.sum(distri_3[3,:]>TH_T2_minp)/NSamples))
    dic = {"SNR":SNR,
            'distri_1':distri_1,
            'distri_2':distri_2,
            'distri_3':distri_3,
            'nl':nl,
            'alpha':alpha,
            'Nx':Nx}
    pickle.dump(dic,open("tab1.pkl","wb"))



def fig9a(nl=32,SNR=0.004,Nx=700,Ny=700,NSamples=50000,N_eval=20000,alpha=1E-1,file=None):
    print("------------Figure 9a-----------")
    if file is None:
        sigma = np.sqrt(2/SNR)
        COV2 = my_cov(nl,sigma=sigma,slope=-0.1,init=1)
        betas,Ns = loss(COV2,nl,Nx,Ny,NSamples=NSamples,alpha=alpha,N_eval=N_eval,
                    getTtest=True,getDtest=False)
    else:
        dic = pickle.load(open(file,"rb"))
        SNR = dic["SNR"]
        betas = dic["betas"]
        Ns = dic["Ns"]
        Nx = dic["Nx"]
        COV2 = dic["COV"]

    plt.figure()
    plt.semilogy(2*Ns,betas[0,:],label=r"incorrect threshold",color=colors["Ttest"])
    plt.semilogy(2*Ns,betas[1,:],label=r"correct threshold",ls="-.",color=colors["Ttest"])
    plt.grid(True,which="both",ls="-")
    plt.legend()
    plt.xlabel(r"$N$")
    plt.ylabel(r"$\beta$")
    plt.xlim((2*Ns[0],2*Ns[np.where(betas[0]!=0)[0][-1]]))
    plt.savefig("fig9a.pdf",bbox_inches = 'tight', pad_inches = 0)

    dic = {"Ns":Ns,
    'betas':betas,
    'COV':COV2,
    'nl':nl,
    'Nx':Nx,
    'SNR':SNR}

    pickle.dump(dic,open("fig9a.pkl","wb"))
    return betas,Ns

def fig9b(nl=32,SNR=0.004,Nx=700,Ny=700,NSamples=50000,N_eval=20000,alpha=1E-1,file=None):

    print("------------Figure 9b-----------")
    if file is None:
        sigma = np.sqrt(2/SNR)
        COV2 = my_cov(nl,sigma=sigma,slope=-0.1,init=1)
        betas,Ns = loss(COV2,nl,Nx,Ny,alpha=alpha,N_eval=N_eval,NSamples=NSamples,
                        getTtest=False,getDtest=True)
    else:
        dic = pickle.load(open(file,"rb"))
        SNR = dic["SNR"]
        betas = dic["betas"]
        Ns = dic["Ns"]
        Nx = dic["Nx"]
        COV2 = dic["COV"]

    # plot results
    plt.figure()
    plt.semilogy(2*Ns,betas[0,:],label=r"incorrect threshold",color=colors["Dtest"])
    plt.semilogy(2*Ns,betas[1,:],label=r"correct threshold",ls="-.",color=colors["Dtest"])
    plt.grid(True,which="both",ls="-")
    plt.legend()
    plt.xlabel(r"$N$")
    plt.ylabel(r"$\beta$")
    plt.xlim((2*Ns[0],2*Ns[np.where(betas[0]!=0)[0][-1]]))
    plt.savefig("fig9b.pdf",bbox_inches = 'tight', pad_inches = 0)

    dic = {"Ns":Ns,
    'betas':betas,
    'COV':COV2,
    'nl':nl,
    'Nx':Nx,
    'SNR':SNR}
    pickle.dump(dic,open("fig9b.pkl","wb"))


def fig13a(nls=np.linspace(50,200,50,dtype=int),Nx=500,Ny=500,Nav=100,getp=False,file=None):

    print("------------Figure 13a-----------")
    if file is None:
        times = timeVsnl(nls=nls,Nx=Nx,Ny=Ny,Nav=Nav,getp=getp)
    else:
        dic = pickle.load(open(file,"rb"))
        times = dic["times"]
        nls = dic["nls"]
        Nav = dic["Nav"]
        getp = dic["getp"]
        Nx = dic["Nx"]
        Ny = dic["Ny"]


    plt.figure()
    plt.plot(nls,times[0,:],color=colors["Ttest"],label=labels["Ttest"])
    plt.plot(nls,times[1,:],color=colors["Hotelling"],label=labels["Hotelling"])
    plt.plot(nls,times[2,:],color=colors["Dtest"],label=labels["Dtest"],ls=":")
    plt.legend()
    plt.xlabel(r"\huge $n_l$")
    plt.grid(True,which="both",ls="-")
    plt.ylabel('CPU time [s]')
    plt.savefig("fig13a.pdf",bbox_inches = 'tight', pad_inches = 0)


    dic = {"Nx":Nx,
    'nls':nls,
    'times':times,
    'getp':getp,
    'Nav':Nav,
    'Ny':Ny}
    pickle.dump(dic,open("fig13a.pkl","wb"))

def fig13b(Ns=np.linspace(50,500,50,dtype=int),Nav=100,getp=False,nl=200,file=None):
    print("------------Figure 13b-----------")
    if file is None:
        times = timeVsN(Ns=Ns,Nav=Nav,getp=getp,nl=nl)
    else:
        dic = pickle.load(open(file,"rb"))
        times = dic["times"]
        nl = dic["nl"]
        Nav = dic["Nav"]
        getp = dic["getp"]
        Ns = dic["Ns"]

    plt.figure()
    plt.plot(2*Ns,times[0,:],color=colors["Ttest"],label=labels["Ttest"])
    plt.plot(2*Ns,times[1,:],color=colors["Hotelling"],label=labels["Hotelling"])
    plt.plot(2*Ns,times[2,:],color=colors["Dtest"],label=labels["Dtest"],ls=":")
    plt.legend()
    plt.xlabel(r"$N$")
    plt.grid(True,which="both",ls="-")
    plt.ylabel('CPU time [s]')
    plt.savefig("fig13b.pdf",bbox_inches = 'tight', pad_inches = 0)


    dic = {"Ns":Ns,
    'nl':nl,
    'times':times,
    'getp':getp,
    'Nav':Nav}
    pickle.dump(dic,open("fig13b.pkl","wb"))


def fig14(nl=32,SNR=0.01,Nx=2000,Ny=2000,NSamples=50000,N_eval=20000,alpha=1E-5,file=None):

    print("------------Figure 14/15-----------")
    if file is None:
        sigma = np.sqrt(2/SNR)
        COV1 = np.identity(nl)*(2/SNR)
        betas_r,betas_f,betas_f_2,betas_f_3,Ns = compare_fix_vs_random(COV1,nl,Nx,NSamples,alpha=alpha)
    else:
        dic = pickle.load(open(file,"rb"))
        SNR = dic["SNR"]
        betas_r = dic["betas_r"]
        betas_f = dic["betas_f"]
        betas_f_2 = dic["betas_f_2"]
        betas_f_3 = dic["betas_f_3"]


        Ns = dic["Ns"]
        Nx = dic["Nx"]
        COV1 = dic["COV"]

    # plot results
    plt.figure()
    plt.semilogy(2*Ns,betas_f[1,:],label=r"fixed vs fixed",color=colors["Dtest"])
    plt.semilogy(2*Ns,betas_r[1,:],label=r"fixed vs random",ls="-.",color=colors["Dtest"])
    plt.grid(True,which="both",ls="-")
    plt.legend()
    plt.xlabel(r"$N$")
    plt.ylabel(r"$\beta$")
    plt.ylim((1E-2,1))
    plt.savefig("fig14a.pdf",bbox_inches = 'tight', pad_inches = 0)

    # plot results
    plt.figure()
    plt.semilogy(2*Ns,betas_f[0,:],label=r"fixed vs fixed",color=colors["Ttest"])
    plt.semilogy(2*Ns,betas_r[0,:],label=r"fixed vs random",ls="-.",color=colors["Ttest"])
    plt.grid(True,which="both",ls="-")
    plt.legend()
    plt.xlabel(r"$N$")
    plt.ylabel(r"$\beta$")
    plt.ylim((1E-2,1))
    plt.savefig("fig14b.pdf",bbox_inches = 'tight', pad_inches = 0)

    # plot results
    plt.figure()
    plt.semilogy(2*Ns,betas_f[0,:],label=r"Welch's $t$-test with min-$p$, $th_1$",color=colors["Ttest"],ls="-")
    plt.semilogy(2*Ns,betas_f_2[0,:],label=r"$2\times$-Welch's $t$-test with min-$p$, $th_1$",color=colors["Ttest"],ls="-.")
    plt.semilogy(2*Ns,betas_f_3[0,:],label=r"$2\times$-Welch's $t$-test with min-$p$, $th_2$",color=colors["Ttest"],ls="--")
    plt.grid(True,which="both",ls="-")
    plt.legend()
    plt.xlabel(r"$N$")
    plt.ylabel(r"$\beta$")
    plt.ylim((1E-2,1))
    plt.savefig("fig14c.pdf",bbox_inches = 'tight', pad_inches = 0)

    dic = {"Ns":Ns,
    'betas_r':betas_r,
    'betas_f':betas_f,
    'betas_f_2':betas_f_2,
    'betas_f_3':betas_f_3,

    'COV':COV1,
    'nl':nl,
    'Nx':Nx,
    'SNR':SNR}
    pickle.dump(dic,open("fig14.pkl","wb"))



def tab3(nl=32,SNR=0.01,Nx=2000,Ny=2000,NSamples=50000,file=None,alpha=0.1):

    print("------------ tab3 -----------")
    if file is None:
        sigma = np.sqrt(2/SNR)
        COV1 = np.identity(nl)*(2/SNR)
        COV2 = my_cov(nl,sigma=sigma,slope=-0.1,init=1)
        COV3 = my_cov(nl,sigma=sigma,slope=-0.02,init=1)

        alphaTH = 1-(1-alpha)**(1/nl)
        TH = np.abs(scipy.stats.t.ppf((1-alphaTH/2),1/((1/Nx)+(1/Ny))))

        alpha_a = np.sqrt(alpha)
        alphaTH = 1-(1-alpha_a)**(1/nl)
        adjusted_TH_Ttest = np.abs(scipy.stats.t.ppf((1-alphaTH/2),1/((2/Nx)+(2/Ny))))

        alpha_TVLA = np.zeros(3)
        alpha_orig = np.zeros(3)
        alpha_uncor = np.zeros(3)

        alpha_orig[0],alpha_TVLA[0],_,alpha_uncor[0]= compare_TVLA_original(COV1,nl,Nx,NSamples,TH=TH,TH_cor=adjusted_TH_Ttest)
        alpha_orig[1],alpha_TVLA[1],_,alpha_uncor[1]= compare_TVLA_original(COV2,nl,Nx,NSamples,TH=TH,TH_cor=adjusted_TH_Ttest)
        alpha_orig[2],alpha_TVLA[2],_,alpha_uncor[2]= compare_TVLA_original(COV3,nl,Nx,NSamples,TH=TH,TH_cor=adjusted_TH_Ttest)

    else:
        dic = pickle.load(open(file,"rb"))
        SNR = dic["SNR"]
        alpha_orig = dic["alpha_orig"]
        alpha_TVLA = dic["alpha_TVLA"]
        alpha_uncor = dic["alpha_TVLA"]

        Ns = dic["Ns"]
        Nx = dic["Nx"]
        COV1 = dic["COV"]

    print("Estimated alpha (%0.3f): COV1 -  COV2 -  COV3"%(alpha))
    print("Ttest no adjusted threshold :   %0.3f  -    %0.3f   - %0.3f "%(alpha_orig[0],alpha_orig[1],alpha_orig[2]))
    print("Ttest multiple inputs TVLA,adjusted :   %0.3f  -    %0.3f   - %0.3f "%(alpha_TVLA[0],alpha_TVLA[1],alpha_TVLA[2]))
    print("Ttest multiple inputs TVLA,unadjusted:   %0.3f  -    %0.3f   - %0.3f "%(alpha_uncor[0],alpha_uncor[1],alpha_uncor[2]))


    dic = {
    'alpha_orig':alpha_orig,
    'alpha_TVLA':alpha_TVLA,
    'alpha_uncor':alpha_uncor,
    'COV':COV1,
    'nl':nl,
    'Nx':Nx,
    'SNR':SNR}
    pickle.dump(dic,open("tab3.pkl","wb"))


if __name__ == '__main__':

    if len(sys.argv) == 1:
        print("usage: python3 figures.py option")
        print(usage)
        exit(-1)

    fast = False
    paper = False
    use_file = False

    for flag in sys.argv[1:]:
        if flag == 'fast':
            fast = True
        elif flag == 'paper':
            paper = True
        elif flag == 'file':
            use_file = True
        else:
            print("Unknown option")
            print(usage)
            exit(-1)

    if fast:
        print("Running fast mode")
        fig5a(Nav=1)
        fig5b(Nav=1)
        fig6(Nav=1)
        fig8(NSamples=1000)
        tab1(NSamples=1000)
        fig9a(NSamples=1000,N_eval=500)
        fig9b(NSamples=1000,N_eval=500)
        fig13a(Nav=1)
        fig13b(Nav=1)
        fig14(Nav=100)
        tab3(NSamples=1000)

    if paper:
        print("Running paper mode")
        fig5a()
        fig5b()
        fig6()
        fig8()
        tab1()
        fig9a()
        fig9b()
        fig13a()
        fig13b()
        fig14()
        tab3()

    if use_file:
        print("Running file mode")
        fig5a(file="fig5a.pkl")
        fig5b(file="fig5b.pkl")
        fig6(file="fig6.pkl")
        fig8(file="fig8.pkl")
        tab1(file='tab1.pkl')
        fig9a(file="fig9a.pkl")
        fig9b(file="fig9b.pkl")
        fig13a(file="fig13a.pkl")
        fig13b(file="fig13b.pkl")
        fig14(file="fig14.pkl")
        tab3(file='tab3.pkl')
