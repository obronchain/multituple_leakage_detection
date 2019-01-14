import matplotlib.pyplot as plt

SMALL_SIZE = 13
MEDIUM_SIZE = 15
BIGGER_SIZE = 18

labels = {"Ttest":r"Welch + min-$p$",
            "Dtest":"D-test",
            "Hotelling":"Hotelling's $T^2$-test",
            "Cov1":r"\huge $\Sigma_1$",
            "Cov2":r"\huge $\Sigma_2$",
            "Cov3":r"\huge $\Sigma_3$"}

colors = {"Ttest":(0.9,.6,0),
            "Dtest":(.35,.7,.9),
            "Hotelling":(0,.6,.6),
            "Cov1":"r",
            "Cov2":"g",
            "Cov3":"b"}

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('lines',linewidth=2.5)
