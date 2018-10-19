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
data = pd.read_csv(
    data_dir + 'param-no-preferential-pathway.csv',
    header=4,
)


V = 200.0
data = data[(data.SB == 1) & (data.chi == 0)]
n = interp1d(data.p.values, data.n.values)



ps = np.arange(-10,10)
Aes = np.arange(0.1, 1.6, 0.1)
Aes /= 3600.0


p, Ae, c = [], [], []
for Ae_now in Aes:
    for p_now in ps:
            Ae.append(Ae_now*3600.0)
            p.append(p_now)
            c.append(n(p_now)/V/Ae_now)



df = pd.DataFrame(
    {'p': p,
    'Ae': Ae,
    'c': c,}
)

df.c = df.c.apply(np.log10)

ref_state = ((df.p == -5) & (df.Ae == 0.1))

#print(df.c - df.c[ref_state].values)

print(df[ np.abs(df.c - df.c[ref_state].values) >= 1.0 ].dropna())

#print(df.c.apply(np.log10))

"""
data = [
    go.Contour(
        z=df.c.apply(np.log10),
        x=df.p,
        y=df.Ae,
    )]



fig = go.Figure(
    data=data,
    #layout=layout,
)
filename = 'p-ae-contour'
pio.write_image(
    fig,
    file = figures_dir + filename + '.pdf',
    )

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
"""
