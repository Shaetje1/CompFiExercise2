import numpy as np
from scipy.stats import norm
#from Gaussian import gaussian_cf, sin_coeff


S0= 36
sigma = 0.2
r = 0.06
T = 1
K = 40



# exact price - Black-Scholes formula
d1 = ( np.log(S0/K) + (r + 0.5*sigma**2) * T ) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
bs_put = -1 * norm.cdf(-d1)*S0 + norm.cdf(-d2)*K*np.exp(-r*T)


# forward finite difference 
def FTCS(S0,sigma,T):
    L = 20
    Nx = 1000
    Nt = 40000
    h = L/Nx #Space Discretization
    k = T/Nt #Time Discretization
    
    T1  = np.diag([1]* (Nx-2), 1) - np.diag([1] * (Nx-2), -1)
    T2  = -2 * np.diag([1] * (Nx-1)) + np.diag([1]* (Nx-2), 1) + np.diag([1] * (Nx-2), -1)
    
    F = (1 - r*k) * np.diag([1] * (Nx-1))  + 0.5 *k * (sigma**2) /(h**2) * T2 +  k * (r-0.5*(sigma**2))/(2*h) * T1
    
    mvec = np.linspace(start = -L/2 + h, stop = L/2-h, num=Nx-1)
    U = np.zeros((Nx-1, Nt+1))
    U[:, 0] = np.maximum(K - np.exp(mvec), 0)
    
    for i in range(Nt):
        time2mat = i*k
        p = np.zeros(Nx-1)
        p[0] = ( 0.5 *k * (sigma**2) /(h**2) - k * (r-0.5*(sigma**2))/(2*h) ) * K* np.exp(-r*time2mat)
        U[:, i+1] = np.dot(F, U[:, i]) + p
        
    ftcs_price = np.interp(np.log(S0), mvec, U[:, Nt])

    print(f'forward finite difference price is {ftcs_price:<8.4f}')
S0_Values=[36,38,40,42,44]
Vol_values=[0.2,0.4]
T_values=[1,2]

for s in S0_Values:
    for vol in Vol_values:
        for t in T_values:
            FTCS(s,vol,t)
