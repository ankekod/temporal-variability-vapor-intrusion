import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import LSODA
from scipy.interpolate import interp1d
import sqlite3
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import plotly.io as pio

data_dir = './data/preferential-pathway-sensitivity/'
figures_dir = './figures/transient-response/'

# parameters
V = 200.0
df = pd.read_csv('./data/soils/all-data.csv')

Ae = 0.5

t1, t2 = [], []
dc1, dc2 = [], []
Aes = []
dcdt1, dcdt2 = [], []
soils = []
fig, (ax1, ax2) = plt.subplots(1,2)

df.n *= 3600.0
for Ae in [0.5, 1.0, 0.1]:

    for soil in df['soil_type'].unique():
        ref = df[ (df['soil_type'] == soil) & ( df.p==-5 ) ]
        target = df[ (df['soil_type'] == soil) & ( df.p==5 ) ]
        Aes.append(Ae)
        soils.append(soil)
        # unsteady cstr method
        def dudt(t, u):
            dudt =  n/V - u*Ae
            return dudt
        #retrives processed data
        # time variables
        t0 = 0.0 # initial time
        tau = 240 # max allowed time
        # initial/reference concentration
        y0 = ref.n/V/Ae
        for n, p in zip(target.n, target.p):
            # solving for target state change in variable values
            #print('Case: p = %1.1f, Ae = %1.1f' % (p, Ae))
            c = n/V/Ae
            solver = LSODA(
                dudt,
                t0,
                y0.values,
                tau,
                max_step=0.1,
            )
            t, y = [], [] # storage lists
            while solver.y > 1.01*c:
                t.append(solver.t)
                y.append(solver.y)
                try:
                    solver.step()
                except:
                    print('Solver failed in first loop at t = %1.1f' % solver.t)
                    break
            t_eq = solver.t
            t1.append(t_eq)
            dc1.append((c-y0))
            dcdt1.append((c-y0)/t_eq)
            print('Min. reached after %2.1f hours' % t_eq)
            # going back to reference state variables
            n = ref.n
            while solver.y < 0.99*y0.values:
                t.append(solver.t)
                y.append(solver.y)
                try:
                    solver.step()
                except:
                    print('Solver failed in second loop at t = %1.1f' % solver.t)
                    break
            t_org = solver.t - t_eq
            t2.append(t_org)
            dc2.append((y0-c))
            dcdt2.append((y0-c)/t_org)
            print('Max. reached after %2.1f hours' % t_org)


for var in [soils,Aes,t1,t2,dc1,dc2,dcdt1,dcdt2]:
    print(len(var))

df2 = pd.DataFrame(
    {'soil': soils,
    'Ae': Aes,
    't1': t1,
    't2': t2,
    'dc1': dc1,
    'dc2': dc2,
    'dcdt1': dcdt1,
    'dcdt2': dcdt2}
)

df2.plot(x='soil',y='t1',ax=ax1, label='Ae = %1.1f' % Ae)
df2.plot(x='soil',y='t2',ax=ax2,  label='Ae = %1.1f' % Ae)

plt.legend() # implementera soil type, dc = x 
plt.show()
