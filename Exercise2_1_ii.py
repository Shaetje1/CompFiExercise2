# -*- coding: utf-8 -*-
"""
Created on Mo May 18 10:10:21 2026

@author: shaew
"""
#I copy pasted the code from exercise i), i will now make adjustments where necessary to make it work for the MC paths
#Turned out i made some mistakes in exercise i), so it is not that expandable
import numpy as np
from numpy.polynomial.laguerre import lagfit, lagval
#I should later use this variables inside a function to easily make the desired table
#For now i do it mostly manually to get a proof of concept
Strike=40
r=0.06
T=1
dt=T/50
Vol=0.2
S0=36
R=50000


def CashFlow(StockValues):#input the StockValues at a specific time
    SV=StockValues
    CashFlow = np.maximum(Strike-SV,0)
    return CashFlow #Output the cashflow of exercising the option if ITM

def itm_indices(ValuesX):
    itm_indices=np.where(ValuesX<Strike)[0]
    return itm_indices

def SIM(Strike,r,T,dt,Vol,S0,R):
    
    RV=np.random.standard_normal((R,50*T))
    RVantithetic=-RV
    RV=np.vstack((RV,RVantithetic))
    drift=(r-0.5*Vol**2)*dt
    diffusion=Vol*np.sqrt(dt)*RV
    StockData=S0*np.exp(np.cumsum(drift+diffusion,axis=1))
    TimeSeries=StockData.T
    
    DecisionMatrix=np.zeros((50*T,2*R))
    DecisionMatrix[-1]=np.array([1]*2*R)
    cashflows=np.maximum(Strike-TimeSeries[-1],0)
    
    
    for i in range(0,50*T-1):
        #For every timestep we need to
        #Get the itm indices of the earlier time
        #Compare the discounted cashflow value
        #Where the cashflow will get updated to either be discounted once more at each step
        #Or is the value of exercising now
        #Depending on what we find to be optimal using our regression
        cashflows=cashflows*np.exp(-r*dt)
        X=TimeSeries[-2-i]
        itm=itm_indices(X)
        Y=cashflows[itm]
        #We use the built in numpy library for laguerre regression
        #This library doesnt use the weighted version as in the paper
        #So we add in these weights and then reweight them later
        weights=np.exp(-0.5*X[itm]/Strike)
        coeffs=lagfit(X[itm]/Strike,Y/weights,deg=3)
        Continuation=lagval(X[itm]/Strike,coeffs)*weights
        Exercise= Strike-X[itm]
        IndicesToExercise=itm[Exercise>Continuation] #Since itm is a list of indices
        cashflows[IndicesToExercise]=Exercise[Exercise>Continuation]
        
        DecisionMatrix[-2-i,IndicesToExercise]=1
        DecisionMatrix[-1-i:,IndicesToExercise]=0
        
    #Antithetic sampling so the unbiased estimator is the sum divided by 2
    #Where we discount 1 more time to get the value at time 0
    original_part = cashflows[:R]*np.exp(-r*dt)
    antithetic_part = cashflows[R:]*np.exp(-r*dt)
    american=np.mean((original_part+antithetic_part)/2)
    
    european_payoffs = np.maximum(Strike - TimeSeries[-1], 0) * np.exp(-r * T)
    european = np.mean((european_payoffs[:R] + european_payoffs[R:]) / 2)
    
    SE=np.std((original_part+antithetic_part)/2)/np.sqrt(R)
    return (american,european,SE)
  
    
    
header = f"{'S':<4} {'sigma':<6} {'T':<4} | {'American':<10} {'(s.e.)':<8} {'European':<10} {'Premium':<10}"
hline = "-" * len(header)

print(hline)
print(header)
print(hline) 
S0_Values=[36,38,40,42,44]
Vol_values=[0.2,0.4]
T_values=[1,2]

for s in S0_Values:
    for vol in Vol_values:
        for t in T_values:
            am,eu,se=SIM(40,0.06,t,0.02,vol,s,50000)
            prem=am-eu
            print(f"{s:<4} {vol:<6.2f} {t:<4} | {am:<10.4f} {se:<8.4f} {eu:<10.4f} {prem:<10.4f}")
    print('\n')    
            
    
    



    
