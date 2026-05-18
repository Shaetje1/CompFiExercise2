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
paths=len(TimeSeries[-1])
print("      Cash-flow matrix at time 3")
print(f"{'Path':<8} {'t = 1':<10} {'t = 2':<10} {'t = 3':<10}")
print("-"*38)
for i, val in enumerate(CashFlow(TimeSeries[-1]), start=1):
    print(f"{i:<8} {'—':<10} {'—':<10} {val:<10.5f}")
    
#Regression at time 2
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
print(f"Regression Equation time 2: {coeffs[0]:.5f}x² {coeffs[1]:+.5f}x {coeffs[2]:+.5f}")

#Optimal early exercise decision at time 2
paths=len(TimeSeries[-1])
Exercise=np.maximum(Strike-X,0)
Continuation = np.polyval(coeffs,X)
print(f"{'Optimal early exercise decision at time 2':^35}")
print(f"{'Path':<8} {'Exercise':<18} {'Continuation':<8}{'Exercise or not?':>20}")
print("-" * 38)
ExerciseAtT2=[]
for i in range(0,paths):
   
    if i in itm_idx:
        ExerciseAtT2.append(Exercise[i]>Continuation[i])
        print(f"{i+1:<8}{Exercise[i]:<18.5f}{Continuation[i]:<8.5f}{ExerciseAtT2[i]:>20}")
    else:
        print(f"{i+1:<8}{'---':<18}{'---':<18}")
        ExerciseAtT2.append(0)
        
#Cash Flow Matrix at time 2
paths=len(TimeSeries[-1])
print("      Cash-flow matrix at time 2")
print(f"{'Path':<8} {'t = 1':<10} {'t = 2':<10} {'t = 3':<10}")
print("-"*38)
for i in range(paths):
    Bool=Exercise[i]>Continuation[i]
    print(f"{i+1:<8}{'---':<10}{Exercise[i]*Bool:<10.5f}{max(1.1-TimeSeries[-1][i],0)*(1-Bool):>10.5f}")

#Regression at time 1
print(f"{'Regression at time 1':^35}")
print(f"{'Path':<8} {'Y':<18} {'X':<8}")
print("-" * 38)
Y,X=Discount(CashFlow(TimeSeries[-2]),TimeSeries[-3])
itm_idx2=itm(Y,X)
for i in range(len(Y)):
    if Exercise[i]<Continuation[i]:
        Y[i]=0
for i in range(len(TimeSeries[-1])):
    if i in itm_idx2:
        print(f"{i+1:<8}{Y[i]:<18.5f}{X[i]:<8.5f}")
    else:
        print(f"{i+1:<8}{'---':<18}{'---':<18}")
        
#We will now use y,x to denote only the non zero indices
y,x=Y[itm(Y,X)],X[itm(Y,X)]
coeffs= np.polyfit(x,y,deg=2)

#The least squares approximation formula for E[Y|X] at time 1"
print(f"Regression Equation time 1: {coeffs[0]:.5f}x² {coeffs[1]:+.5f}x {coeffs[2]:+.5f}")


#Optimal early exercise decision at time 1
paths=len(TimeSeries[-1])
Exercise=np.maximum(Strike-X,0)
Continuation = np.polyval(coeffs,X)
print(f"{'Optimal early exercise decision at time 1':^35}")
print(f"{'Path':<8} {'Exercise':<18} {'Continuation':<8}{'Exercise or not?':>20}")
print("-" * 38)
ExerciseAtT1=[]
for i in range(paths):

    if i in itm_idx2:
        ExerciseAtT1.append(Exercise[i]>Continuation[i])
        print(f"{i+1:<8}{Exercise[i]:<18.5f}{Continuation[i]:<8.5f}{ExerciseAtT1[i]:>20}")
    else:
        print(f"{i+1:<8}{'---':<18}{'---':<18}")
        ExerciseAtT1.append(0)
        
        
#Stopping Rule
ExerciseAtT3=[]
for i in TimeSeries[-1]:
    ExerciseAtT3.append(i<Strike)
print(f"{'Naive Stopping rule':^38}")
print(f"{'Path':<8}{'t=1':<10}{'t=2':<10}{'t=3':<10}")
for i in range(paths):
    print(f"{i+1:<8}{ExerciseAtT1[i]:<10}{ExerciseAtT2[i]:<10}{ExerciseAtT3[i]:<10}")
print("An obvious issue here is that we are exercising some options at more than 1 timepoint, which is not possible")
for i in range(paths):
    if ExerciseAtT1[i]==1:
        ExerciseAtT2[i]=ExerciseAtT3[i]=0
    if ExerciseAtT2[i]==1:
        ExerciseAtT3[i]=0
        
#This code is far from scalable, but it works to get the idea, i will have to improve it for the next step though
print(f"{'Improved Stopping rule':^38}")
print(f"{'Path':<8}{'t=1':<10}{'t=2':<10}{'t=3':<10}")
for i in range(paths):
    print(f"{i+1:<8}{ExerciseAtT1[i]:<10}{ExerciseAtT2[i]:<10}{ExerciseAtT3[i]:<10}")

print(f"{'Option cash flow matrix':^38}")
print(f"{'Path':<8}{'t=1':<10}{'t=2':<10}{'t=3':<10}")
Tot=0
for i in range(paths):
    Val1=max(Strike-TimeSeries[-3][i],0)*ExerciseAtT1[i]
    Val2=max(Strike-TimeSeries[-2][i],0)*ExerciseAtT2[i]
    Val3=max(Strike-TimeSeries[-1][i],0)*ExerciseAtT3[i]
    print(f"{i+1:<8}{Val1:<10.5f}{Val2:<10.5f}{Val3:<10.5f}")
    Val4=max(Val1*np.exp(-r),Val2*np.exp(-2*r),Val3*np.exp(-3*r))
    Tot+=Val4
print(f"{'American Option value ':<12}{Tot/paths:<10.5f}")
Tot=0
for i in range(paths):
    Tot+=np.exp(-3*r)*max(0,Strike-TimeSeries[-1][i])
print(f"{'European Option value ':<12}{Tot/paths:<10.5f}")








