import os
import sys
import numpy as np
import pandas as pd
import pickle
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt


class Person:
  def __init__(mysillyobject, name, age):
    mysillyobject.name = name
    mysillyobject.age = age

  def myfunc(abc):
    print("Hello my name is " + abc.name)

## This creates parent class creates an instance of 
## a dynamic system define by Component names (CName)
## parsName = parameter name and time span
class system:
    def __init__(self, CName, parsName, t_span):
        self.CName = CName
        self.parsName = parsName
        self.t_span = t_span
    



# We let C be the vector of components
# We have 9 components in total

# create a named list to keep track of components.
# y is the qoutient rna/rna_min
CName = ["x", "y", "s", "b", "ba", "a", "aa", "e", "c02", "h2"]

# Create a list with the names of parameters of the dynamic system
parsName = [
    "k1", "k2", "k3", "k4", "k5", "k6",
    "k7", "k8", "k9", "k10", "k11", "k12",
    "k13", "k14", "ki", "ks", "kba", "kaa"
]

# This is a constant that is used throughout the program
lambdaConst = 0.56


# Create a list with parameter values.
parsVals = np.array([
    0.0090, 0.0008, 0.0255, 0.6764, 0.0136,
    0.1170, 0.0113, 0.7150, 0.1350, 0.1558,
    0.0258, 0.6139, 0.0185, 0.00013, 0.833,
    2.0, 0.5, 0.5
])


# Now define function that defines the system, the output is the 
# RHS of the first order non-linear ODEs

# Note that the system is an ODE of order 1
# t is a sequence
def system(t, C, pars=parsVals):
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


# Initial values for C
# for glucose (=s) we set the initial value to 50 g/l,
# and the biomass (=x) to 10 g/l
def initialValues(parsVals=parsVals):
    CName = ["x", "y", "s", "b", "ba", "a", "aa", "e", "c02", "h2"]

    # Create variables for initial values contained in initC

    initCsol, initCdot = np.zeros(len(CName)), np.zeros(len(CName))

    xInit = 0.1
    yInit = 1.1
    sInit = 60

    ## Start by initializing initCsol
    initCsol = np.concatenate(
        (
            np.array([xInit, yInit, sInit]),
            np.ones(7)/100
        )
    )

    initCsol[CName.index("aa")] = 0.001

    ## Now initialize initCdot using the mass balance equations.
    # initial deriviative for butyrate ("ba")
    initCdot[CName.index("ba")] = parsVals[parsName.index("k5")]*sInit* \
    (parsVals[parsName.index("ki")]/(parsVals[parsName.index("ki")]+initCsol[CName.index("b")]))*xInit \
    - parsVals[parsName.index("k6")]*(initCsol[CName.index("ba")]/(parsVals[parsName.index("ks")]+initCsol[CName.index("ba")]))*xInit 

    # Initial derivative for butanol
    initCdot[CName.index("b")] = parsVals[parsName.index("k7")]*sInit*xInit - 0.841*initCdot[CName.index("ba")]

    # Initial derivate for acetic acid
    initCdot[CName.index("aa")] = parsVals[parsName.index("k8")]*(initCsol[CName.index("s")]/(parsVals[parsName.index("ks")]+initCsol[CName.index("s")])) \
    * (parsVals[parsName.index("ki")]/(parsVals[parsName.index("ki")]+initCsol[CName.index("b")]))*initCsol[CName.index("x")] \
    - parsVals[parsName.index("k9")]*(initCsol[CName.index("aa")]/(parsVals[parsName.index("kaa")]*initCsol[CName.index("a")]+initCsol[CName.index("aa")])) \
    * (initCsol[CName.index("s")]/(parsVals[parsName.index("ks")]+initCsol[CName.index("s")]))*initCsol[CName.index("x")]

    # initial derivat for acetone ("a")
    initCdot[CName.index("a")] = parsVals[parsName.index("k10")]*(initCsol[CName.index("s")]/(parsVals[parsName.index("ks")]+initCsol[CName.index("s")]))*initCsol[CName.index("x")] \
    - 0.484*initCdot[CName.index("aa")]

    #initial derivative for Ethanol ("e")
    initCdot[CName.index("e")] = parsVals[parsName.index("k11")]*(initCsol[CName.index("s")] /
    (parsVals[parsName.index("ks")]+initCsol[CName.index("s")]))*initCsol[CName.index("x")]

    #initial derivative for c02
    initCdot[CName.index("c02")] = parsVals[parsName.index("k12")]*(initCsol[CName.index("s")] /
    (parsVals[parsName.index("ks")]+initCsol[CName.index("s")]))*initCsol[CName.index("x")]
    
    #initial derivative for h2
    initCdot[CName.index("h2")] = parsVals[parsName.index("k13")]*(initCsol[CName.index("s")]/(parsVals[parsName.index("ks")]+initCsol[CName.index("s")]))*initCsol[CName.index("x")] \
    + parsVals[parsName.index("k14")]*initCsol[CName.index("s")]*initCsol[CName.index("x")]

    ## Now do derivative for biomass (x)
    initCdot[CName.index("x")] = lambdaConst*(initCsol[CName.index("y")]-1)*initCsol[CName.index("x")] \
            - parsVals[parsName.index("k2")]*initCsol[CName.index("x")
                                            ]*initCsol[CName.index("b")]

    ## Now do derivative for substrate/glucose (s)
    initCdot[CName.index("s")] = -parsVals[parsName.index("k3")]*initCsol[CName.index("s")]*initCsol[CName.index("x")] \
            - parsVals[parsName.index("k4")]*(initCsol[CName.index("s")]/(
                parsVals[parsName.index("ks")]+initCsol[CName.index("s")]))*initCsol[CName.index("x")]


    ## Now do derivative for rna/rna_min (y)
    initCdot[CName.index("y")] = (parsVals[parsName.index("k1")]*initCsol[CName.index("s")]
                                * (parsVals[parsName.index("ki")]/(parsVals[parsName.index("ki")]+initCsol[CName.index("b")]))
                                - lambdaConst*(initCsol[CName.index("y")] - 1))*initCsol[CName.index("y")]


    initCdot[CName.index("a")] = 0.1
    return(np.concatenate((initCsol, initCdot)))


## Now call the ODE solver
t_span = np.linspace(0, 240, 240001)

C_sim = odeint(
    func = system,
    t=t_span,
    y0=initialValues()[:10],
    tfirst=True
)
C_sim[0,:]

## Now plot the simulated results

fig, axs = plt.subplots(3,4)

nplot = 0
for i in range(3):
    for j in range(4):
        if nplot <= (len(C_sim[0,:])-1):
            axs[i,j].plot(t_span, C_sim[:,nplot])
            axs[i,j].set_title(CName[nplot])
            nplot = nplot + 1
        else:
            fig.delaxes(axs[i][j])
            nplot = nplot + 1
 

for ax in axs.flat:
    ax.set(xlabel = "Hours", ylabel = "g/L")

plt.tight_layout()
plt.savefig(os.getcwd()+"\Output"+"\system.jpeg")
plt.show()

## So we see that we can simulate from the dynamic systems.
## Now we add noise to the system measurements corresponding
## to 5% rsd.
rsd = 0.03
## Initialize a numpy array with noisy measurements of system.
C_vector = C_sim
## initialize error_vector
error_vector = np.zeros(len(C_sim[1,:]))
cov = np.zeros((len(C_sim[1,:]), len(C_sim[1,:])))
for i in range(len(C_vector)):
    np.fill_diagonal(cov, (C_sim[i,:]*rsd)**2)
    error_vector = np.random.multivariate_normal(
        mean = np.zeros(len(C_sim[0,:])), cov = cov        
    )
    C_vector[i,:] = C_vector[i,:] + error_vector


## Now plot the noisy results

fig, axs = plt.subplots(3,4)

nplot = 0
for i in range(3):
    for j in range(4):
        if nplot <= (len(C_vector[0,:])-1):
            axs[i,j].plot(t_span, C_vector[:,nplot])
            axs[i,j].set_title(CName[nplot])
            nplot = nplot + 1
        else:
            fig.delaxes(axs[i][j])
            nplot = nplot + 1
 

for ax in axs.flat:
    ax.set(xlabel = "Hours", ylabel = "g/L")

plt.tight_layout()
plt.savefig(os.getcwd()+"\Output"+"\\noisysystem.jpeg")
plt.show()



## Finally, for real datasets, we have at most 5 data points for certain parameters
## and one sample per hour

# initialize numpy array containing sparse measurements

C_vector_sparse = np.empty(C_vector.shape)
C_vector_sparse[:] = np.nan

## sampling frequencies
hours = np.linspace(start=0, stop=240, num=241)
days = np.linspace(start=24, stop=240, num=10)[:-1]
bidaily = np.linspace(start=24, stop=240, num=19)[:-1]
# hourly measurements will be of c02 h2, substrate and biomass
hourlyVars = ["c02", "s", "h2", "x"]
hourlyVarsInd = []
for i in range(len(hourlyVars)):
    hourlyVarsInd.append(
        CName.index(hourlyVars[i])
    )

dailyVars = ["a", "b", "e"]
dailyVarsInd = []
for i in range(len(dailyVars)):
    dailyVarsInd.append(
        CName.index(dailyVars[i])
    )

bidailyVars = [x for x in  CName if ((x not in dailyVars) & (x not in hourlyVars)) ]


## Now fill C_vector_sparse
for i in range(len(hourlyVars)):
    rows = np.in1d(t_span, hours)
    cols = CName.index(hourlyVars[i])
    C_vector_sparse[rows,cols] = C_sim[rows,cols]

for i in range(len(dailyVars)):
    rows = np.in1d(t_span, days)
    cols = CName.index(dailyVars[i])
    C_vector_sparse[rows,cols] = C_sim[rows,cols]

for i in range(len(bidailyVars)):
    rows = np.in1d(t_span, bidaily)
    cols = CName.index(bidailyVars[i])
    C_vector_sparse[rows,cols] = C_sim[rows,cols]

## Now plot sparse values
fig, axs = plt.subplots(3,4)

nplot = 0
for i in range(3):
    for j in range(4):
        if nplot <= (len(C_vector_sparse[0,:])-1):
            axs[i,j].plot(t_span, C_vector_sparse[:,nplot], 'o')
            axs[i,j].set_title(CName[nplot])
            nplot = nplot + 1
        else:
            fig.delaxes(axs[i][j])
            nplot = nplot + 1
 

for ax in axs.flat:
    ax.set(xlabel = "Hours", ylabel = "g/L")

plt.tight_layout()
plt.savefig(os.getcwd()+"\Output"+"\\noisysystem_sparse.jpeg")
plt.show()

## Generally the data is received and handled as a dataframe.
## so we translate the numpy array C_vector_sparse to a 
## dataframe data.
CName.insert(0, "Hours")
dfData = pd.DataFrame(
    np.concatenate(
        (t_span.reshape(len(t_span), 1), C_vector_sparse),
        axis = 1
    ),
    columns = CName
)

dfData.to_csv(
    os.getcwd() + "\\Output\\data.csv"
)
