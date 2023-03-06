import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt


## This creates parent class creates an instance of 
## a dynamic system define by Component names (CName)
## parsName = parameter name and time span

## If you want to change the structure of the system, you need
## to change def system_structure function.

class system:
    def __init__(
            self, CName, parsName, parsVals, t_span,
            initialStates
        ):
        self.CName = CName
        self.parsName = parsName
        self.t_span = t_span
        self.parsVals = parsVals
        self.lambdaConst = 0.56
        self.initialStates = initialStates
    
    ## This function defines the mass balance equations 
    ## for the system
    def system_structure(self, t, C):
        pars = self.parsVals
        CName = self.CName
        parsName = self.parsName
        lambdaConst = self.lambdaConst

        Csol, Cdot = C[:int(len(C))], np.empty(len(C))
        # this is the differential equation for the x (biomass)
        Cdot[CName.index("x")] = lambdaConst*(Csol[CName.index("y")]-1)*Csol[CName.index("x")] \
            - pars[parsName.index("k2")]*Csol[CName.index("x")
                                            ]*Csol[CName.index("b")]

        # This is the differential equation for the y (rna/rna_min)
        Cdot[CName.index("y")] = (pars[parsName.index("k1")]*Csol[CName.index("s")]
                                * (pars[parsName.index("ki")]/(pars[parsName.index("ki")]+Csol[CName.index("b")]))
                                - lambdaConst*(Csol[CName.index("y")] - 1))*Csol[CName.index("y")]

        # This is the differential equation for s (glucos substrate)
        Cdot[CName.index("s")] = -pars[parsName.index("k3")]*Csol[CName.index("s")]*Csol[CName.index("x")] \
            - pars[parsName.index("k4")]*(Csol[CName.index("s")]/(
                pars[parsName.index("ks")]+Csol[CName.index("s")]))*Csol[CName.index("x")]

        # This is the differential equation for ba (butyrate)
        Cdot[CName.index("ba")] = pars[parsName.index("k5")]*Csol[CName.index("s")]*(pars[parsName.index("ki")]/(pars[parsName.index("ki")]+Csol[CName.index("b")]))*Csol[CName.index("x")] \
            - pars[parsName.index("k6")]*(Csol[CName.index("ba")]/(
                pars[parsName.index("kba")]+Csol[CName.index("ba")]))*Csol[CName.index("x")]

        # This is the differential equation for b (butanol)
        Cdot[CName.index("b")] = pars[parsName.index(
            "k7")]*Csol[CName.index("s")]*Csol[CName.index("x")]-0.841*Cdot[CName.index("ba")]

        # This is the differential equation for aa (acetic acid)
        Cdot[CName.index("aa")] = pars[parsName.index("k8")]*(Csol[CName.index("s")]/(pars[parsName.index("ks")]+Csol[CName.index("s")])) \
            * (pars[parsName.index("ki")]/(pars[parsName.index("ki")]+Csol[CName.index("b")]))*Csol[CName.index("x")] \
            - pars[parsName.index("k9")]*(Csol[CName.index("aa")]/(pars[parsName.index("kaa")]*Csol[CName.index("a")]+Csol[CName.index("aa")])) \
            * (Csol[CName.index("s")]/(pars[parsName.index("ks")]+Csol[CName.index("s")]))*Csol[CName.index("x")]

        # This is the differential equation for a (acetone production)
        Cdot[CName.index("a")] = pars[parsName.index("k10")]*(Csol[CName.index("s")]/(pars[parsName.index("ks")]+Csol[CName.index("s")]))*Csol[CName.index("x")] \
            - 0.484*Cdot[CName.index("aa")]

        # This is the differential equation for e (ethanol)
        Cdot[CName.index("e")] = pars[parsName.index("k11")]*(Csol[CName.index("s")] /
                                                            (pars[parsName.index("ks")]+Csol[CName.index("s")]))*Csol[CName.index("x")]

        # This is the differential equation for c02
        Cdot[CName.index("c02")] = pars[parsName.index("k12")]*(Csol[CName.index("s")] /
                                                                (pars[parsName.index("ks")]+Csol[CName.index("s")]))*Csol[CName.index("x")]

        # This is the differential equation for h2
        Cdot[CName.index("h2")] = pars[parsName.index("k13")]*(Csol[CName.index("s")]/(pars[parsName.index("ks")]+Csol[CName.index("s")]))*Csol[CName.index("x")] \
            + pars[parsName.index("k14")]*Csol[CName.index("s")
                                            ]*Csol[CName.index("x")]
        
        return (Cdot)        
    
    def system_solve(self):
        C_sim = odeint(
            func = self.system_structure,
            t=self.t_span,
            y0=self.initialStates,
            tfirst=True
        )

        return(C_sim)        

## The next class will inherit from system,
## but here we add measurement noise and a random 
## adjustment to the rate equations.
class system_noise(system):
    def __init__(self, CName, parsName, parsVals, t_span, initialStates, varBatch, varMeasurement, sparse_dict="None"):
        super().__init__(
            CName, parsName, parsVals, t_span, initialStates            
        )
        self.varBatch = varBatch
        self.varMeasurement = varMeasurement
        self.ncomps = len(self.CName)
        self.sparse_dict = sparse_dict
        if isinstance(self.varBatch, float) or isinstance(self.varBatch, int):
            mu = 0
        else:
            mu = np.zeros((len(self.varBatch))) 

        self.rateRandomEffect = np.random.normal(
            mu, 
            np.sqrt(self.varBatch), 
            self.ncomps
        )

        if np.sum(self.varBatch)==0:
            self.rateRandomEffect = np.zeros(len(self.CName))
        


    ## In this case we add the random effects to the rates 
    
    def system_structure_randomeffects(self, t, C):
        Cdot = self.system_structure(t, C) - np.abs(self.rateRandomEffect)
        return(Cdot)
    
    def system_sim(self):
        C_simInit = odeint(
            func = self.system_structure_randomeffects,
            t=self.t_span,
            y0=self.initialStates,
            tfirst=True
        )

        error = np.random.normal(
            C_simInit, np.sqrt(self.varMeasurement)*np.abs(C_simInit)            
        )

        C_sim = C_simInit + error
        return(C_sim)
    
    ## Add a method that generates a sparse dataset.
    # The sparsity is determined by a dictionary, containing {CName : t_span (to keep)}

    def system_sim_sparse(self):
        if self.sparse_dict == "None":
            return(print("No sparsity_dict"))
        
        ## Simulate from noisy system
        sims = self.system_sim()
        
        for i in range(len(self.CName)):
            nanInd = np.isin(
                self.t_span, 
                self.sparse_dict[self.CName[i]],
                invert = True
            )

            sims[nanInd,self.CName.index(self.CName[i])] = np.nan
        
        return(sims)

## Define t_start, t_end and stepsize
t_start = 0
t_end = 48
stepsize = 0.1

## Define the inputs to the class definition

CName=["x", "y", "s", "b", "ba", "a", "aa", "e", "c02", "h2"]
parsName = [
    "k1", "k2", "k3", "k4", "k5", "k6",
    "k7", "k8", "k9", "k10", "k11", "k12",
    "k13", "k14", "ki", "ks", "kba", "kaa"
]

t_span = np.linspace(t_start, t_end, int((t_end - t_start)/stepsize + 1))
parsVals = np.array([
    0.0090, 0.0008, 0.0255, 0.6764, 0.0136,
    0.1170, 0.0113, 0.7150, 0.1350, 0.1558,
    0.0258, 0.6139, 0.0185, 0.00013, 0.833,
    2.0, 0.5, 0.5
])
initialValues = np.concatenate(
    (
        np.array([10, 1.1, 30]),
        np.ones(
            len(CName) - 3,
        )/100
    )
)

initialValues[CName.index("a")] = 1

## number of batches to simulate
n_batch = 10

## Variances
varMeasurement = 0.001
varBatch = np.concatenate(
    (
        np.array([0.05, 0.02, 0.2])/100,
        np.zeros(len(CName)-3)
    )
)

        
## Create the dictionary that defines the sparse sampling
tlist = list()
for i in range(len(CName)):
    ## For x, y, s we sample every hour
    if i <= 2:
        tlist.append(
            np.linspace(t_start, t_end, int((t_end - t_start)/1 + 1))
        )
    ## For these parameters we sample every 4 hours
    elif i<=5:
        tlist.append(
            np.linspace(t_start, t_end, int((t_end - t_start)/4 + 1))
        )
    ## For these parameters we sample every 4 hours
    else:
        tlist.append(
            np.linspace(t_start, t_end, int((t_end - t_start)/12 + 1))
        )

sparse_dict = {CName[i]: tlist[i] for i in range(len(CName))}
sparse_dict


## Creata a plotting function

def plotFermentation(
        classtype = "system",
        CName = CName,
        parsName = parsName,
        parsVals = parsVals,
        t_span = t_span,
        n_batch = 10,
        initialStates = initialValues,
        varBatch = varBatch,
        varMeasurement = varMeasurement,
        sparse_dict = sparse_dict
    ):
    
    ## Create an instance of system object.

    batches = list()
    allsims = np.empty((0, len(CName)))

    for i in range(n_batch):
        if classtype == "system":
            bio_sims = system(
                CName = CName,
                parsName = parsName,
                parsVals = parsVals,
                t_span = t_span,
                initialStates = initialStates
            ).system_solve()
        elif classtype == "noisy system":
            bio_sims = system_noise(
                CName = CName,
                parsName = parsName,
                parsVals = parsVals,
                t_span = t_span,
                initialStates = initialStates,
                varBatch = varBatch,
                varMeasurement = varMeasurement
            ).system_sim()
        else:
            if sparse_dict == "None":
                sys.exit("No sparse_dict")
            
            bio_sims = system_noise(
                CName = CName,
                parsName = parsName,
                parsVals = parsVals,
                t_span = t_span,
                initialStates = initialStates,
                varBatch = varBatch,
                varMeasurement = varMeasurement,
                sparse_dict=sparse_dict
            ).system_sim_sparse()



        allsims = np.concatenate(
            (
                allsims,
                bio_sims            
            )
        )
        batches = batches + ["bio" + str(i+1)]*bio_sims.shape[0]

    nrowSP = 3
    ncolSP = 4
    t_span_plot = np.tile(t_span, n_batch).reshape((len(t_span)*n_batch, 1))

    ## Create pandas dataframe
    df = pd.DataFrame(
        np.concatenate(
            (t_span_plot, allsims),
            axis = 1
        ),
        columns = ["Hours"] + CName
    )

    df["Batch"] = batches

    ## Now generate the plots
    fig, axs = plt.subplots(nrowSP,ncolSP)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    nplot = 0
    for i in range(nrowSP):
        for j in range(ncolSP):
            if nplot < allsims.shape[1]:
                # axs[i,j].plot(t_span_plot, allsims[:,i])
                if classtype == "noisy system" or classtype == "system":
                    sns.lineplot(
                        data=df[["Hours", "Batch", CName[nplot]]].dropna(), 
                        x="Hours", 
                        y = CName[nplot],
                        ax = axs[i,j],
                        hue="Batch"
                    )
                else:
                    sns.scatterplot(
                        data=df[["Hours", "Batch", CName[nplot]]].dropna(), 
                        x="Hours", 
                        y = CName[nplot],
                        ax = axs[i,j],
                        hue="Batch", s = 15
                    )
                axs[i,j].set_ylabel(CName[nplot])               
                axs[i,j].set_title(CName[nplot])
                axs[i,j].get_legend().remove()
                nplot = nplot + 1
            else:
                fig.delaxes(axs[i][j])
                nplot = nplot + 1

    handles, labels = axs[1,1].get_legend_handles_labels()
    plt.subplots_adjust(
        left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5
    )
    fig.legend(handles, labels, loc='right')
    return(fig)



## Plot the dynamic system
plotsystem = plotFermentation(
    classtype = "system",
    CName = CName,
    parsName = parsName,
    parsVals = parsVals,
    t_span = t_span,
    n_batch = 10,
    initialStates = initialValues,
    varBatch = varBatch,
    varMeasurement = varMeasurement,
    sparse_dict = "none"    
)

plotsystem.suptitle("Deterministic system")
plotsystem.savefig(
    fname = os.getcwd() + "\\pyscripts\\Output\\system.png"
)

plotsystem.show()


## Plot the dynamic system with noise    

plotsystem = plotFermentation(
    classtype = "noisy system",
    CName = CName,
    parsName = parsName,
    parsVals = parsVals,
    t_span = t_span,
    n_batch = 10,
    initialStates = initialValues,
    varBatch = varBatch,
    varMeasurement = varMeasurement,
    sparse_dict = "none"    
)

plotsystem.suptitle("System with rate variation and noise")
plotsystem.savefig(
    fname = os.getcwd() + "\\pyscripts\\Output\\noisy_system.png"
)

plotsystem.show()


## Plot the sparse samples of dynamic system with noise    

plotsystem = plotFermentation(
    classtype = "sparse noisy system",
    CName = CName,
    parsName = parsName,
    parsVals = parsVals,
    t_span = t_span,
    n_batch = 10,
    initialStates = initialValues,
    varBatch = varBatch,
    varMeasurement = varMeasurement,
    sparse_dict = sparse_dict    
)
plotsystem.suptitle("Sparse samples from noisy system")
plotsystem.savefig(
    fname = os.getcwd() + "\\pyscripts\\Output\\noisy_system_sparse.png"
)

plotsystem.show()


