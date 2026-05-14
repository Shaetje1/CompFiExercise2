# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:29:21 2026

@author: shaew
"""
import numpy as np
Strike=1.1
r=0.06
StockData=np.array([[1,1.09,1.08,1.34],[1,1.16,1.26,1.54],[1,1.22,1.07,1.03],[1,0.93,0.97,0.92],[1,1.11,1.56,1.52],[1,0.76,0.77,0.9],[1,0.92,0.84,1.01],[1,0.88,1.22,1.34]])
TimeSeries=StockData.T
def CashFlow(TerminalValues):
    TV=TerminalValues
    CashFlow = np.maximum(Strike-TV,0)
    return CashFlow

def Discount(ValuesY,ValuesX): #Make sure Y is from a later time than X,
    return np.array([ValuesY*np.exp(-r),ValuesX])

def itm(ValuesY,ValuesX):
    itm_indices=np.where(ValuesX<Strike)[0]
    return itm_indices
    

#Cash Flow Matrix at time 3
#Written to be slightly more general to be able to reuse later, should maybe turn it into a function
timesteps=3
paths=len(TimeSeries[-1])
print("      Cash-flow matrix at time 3")
print(f"{'Path':<8} {'t = 1':<10} {'t = 2':<10} {'t = 3':<10}")
print("-"*38)
for i, val in enumerate(CashFlow(TimeSeries[-1]), start=1):
    print(f"{i:<8} {'—':<10} {'—':<10} {val:<10.5f}")
    
#Regression at time 2
timesteps=3
paths=len(TimeSeries[-1])
print(f"{'Regression at time 2':^35}")
print(f"{'Path':<8} {'Y':<18} {'X':<8}")
print("-" * 38)
#Y is the discounted cashflow received at t=3, if we don't exercise
#X is the stock price at time 2
Y,X=Discount(CashFlow(TimeSeries[-1]),TimeSeries[-2])
itm_idx=itm(Y,X)
for i in range(len(TimeSeries[-1])):
    if i in itm_idx:
        print(f"{i+1:<8}{Y[i]:<18.5f}{X[i]:<8.5f}")
    else:
        print(f"{i+1:<8}{'---':<18}{'---':<18}")
        
#We will now use y,x to denote only the non zero indices
y,x=Y[itm(Y,X)],X[itm(Y,X)]
coeffs= np.polyfit(x,y,deg=2)

#The least squares approximation formula for E[Y|X] at time 2"
print(f"Regression Equation: {coeffs[0]:.5f}x² {coeffs[1]:+.5f}x {coeffs[2]:+.5f}")

#Optimal early exercise decision at time 2
timesteps=3
paths=len(TimeSeries[-1])
Exercise=np.maximum(Strike-X,0)
Continuation = np.polyval(coeffs,X)
print(f"{'Optimal early exercise decision at time 2':^35}")
print(f"{'Path':<8} {'Exercise':<18} {'Continuation':<8}")
print("-" * 38)

for i in range(len(TimeSeries[-1])):
    if i in itm_idx:
        print(f"{i+1:<8}{Exercise[i]:<18.5f}{Continuation[i]:<8.5f}")
    else:
        print(f"{i+1:<8}{'---':<18}{'---':<18}")





