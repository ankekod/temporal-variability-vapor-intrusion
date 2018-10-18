import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import LSODA
import sqlite3
import pandas as pd

# constants
Ae = 0.5 # air exchange rate 1/s
V = 1.0 # basement volume m3
data_dir = './data/preferential-pathway-sensitivity/'


pp = pd.read_csv(
    data_dir + 'param-preferential-pathway.csv',
    header=4,
)

print(pp)

def dudt(t, u):
    dudt =  n(t)/V - u*Ae
    return dudt


n_max, n0 = 10.0, 1.0
y_max, y0 = n_max/V/Ae, n0/V/Ae

t0 = 0.0
tau = 50.0
solver = LSODA(dudt, t0, [y0], tau)

t_eq = 10.0
t, y = [], []
n = lambda t: n_max if (t <= t_eq) else n0
while solver.y < 0.99*y_max:
    t.append(solver.t)
    y.append(solver.y)
    solver.step()
t_eq = solver.t
while solver.y > 1.01*y0:
    t.append(solver.t)
    y.append(solver.y)
    solver.step()
t_org = solver.t


plt.plot(t, y)

plt.plot([t[0], t[-1]], [y_max, y_max], 'k--')
plt.plot([t[0], t[-1]], [y0, y0], 'k--')

plt.show()
