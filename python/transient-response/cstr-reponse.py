import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sqlite3
import pandas as pd
"""
# constants
Ae = 0.5/3600.0 # air exchange rate 1/s
V = 300.0 # basement volume m3
D = 7.4e-6 # diffusivity of TCE in air m2/s
L = 15.0e-2 # foundation slab thickness m

# variables
w = lambda t: 10.0**(-6.0 + 2.0*np.sin(2.0*np.pi*t/period))
c = 1.0e-2 # concentration assumed constant

# functions
Pe = lambda w: w*L/(2.0*D)
n = lambda w, c: w*c/(1.0 - np.exp(-Pe(w)))

def dudt(u, t):

    w = 10.0**(-6.0 + 2.0*np.sin(2.0*np.pi*t/period))
    Pe = w*L/(2.0*D)
    n = w*c/(1.0 - np.exp(-Pe))
    dudt =  n/V - u*Ae
    return dudt

#dudt = lambda u, t: n(w,c)/V - u*Ae


# parameters
period = 3600.0 # 1 hour period

t = np.linspace(0,period)


u0 = n(w(0), c)/V/Ae


sol = odeint(dudt, u0, t)
print(sol)
"""
