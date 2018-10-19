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
n_func = interp1d(data.p.values, data.n.values)



ps = np.arange(-10,11)
Aes = np.arange(0.1, 1.6, 0.1)
Aes /= 3600.0


p, Ae, c, n = [], [], [], []
for Ae_now in Aes:
    for p_now in ps:
            Ae.append(Ae_now*3600.0)
            p.append(p_now)
            c.append(n_func(p_now)/V/Ae_now)
            n.append(n_func(p_now))



df = pd.DataFrame(
    {
    'p': p,
    'Ae': Ae,
    'c': c,
    'n': n,
    }
)


#df.c = df.c.apply(np.log10)

ref_state = ((df.p == -5) & (df.Ae == 0.1))
ref = df[ref_state].copy()

target = df[(ref.c.values/df.c > 10.0) & (df.Ae <= 1.0) & (df.p < 0)].copy()


target.Ae /= 3600.0
ref.Ae /= 3600.0
def dudt(t, u):
    dudt =  n/V - u*Ae
    return dudt

t0 = 0.0
tau = 128*3600.0
t_eq = tau

y0 = ref.n/V/ref.Ae
print(y0)
for n, Ae, p in zip(target.n, target.Ae, target.p):
    #Ae = lambda t: Ae_max if (t <= t_eq) else Ae0
    #print(n, Ae, p, c)
    c = n/V/Ae
    print(c)
    solver = LSODA(
        dudt,
        t0,
        y0.values,
        tau,
        max_step=60.0,
    )
    t, y = [], []
    while solver.y > 1.01*c:
        t.append(solver.t)
        y.append(solver.y)
        try:
            solver.step()
        except:
            print('Solver failed in first loop.')
            break
    t_eq = solver.t
    print(t_eq)
    #print('Min. reached after %2.1f hours' % float(t_eq)/3600.0)
    Ae = ref.Ae.values

    while solver.y < 0.99*y0.values:
        t.append(solver.t)
        y.append(solver.y)
        try:
            solver.step()
        except:
            print('Solver failed in second loop.')
            break
    t_org = solver.t
    plt.semilogy(np.array(t)/3600.0, y, label='p = %i, Ae = %1.1f' % (p, Ae*3600.0*10.0))

plt.legend()
plt.show()



"""




n_max, n0 = n(-5), n(-5)
Ae_max, Ae0 = 1.0/3600.0, 0.1/3600.0
y_max, y0 = n_max/V/Ae_max, n0/V/Ae0

t, y = [], []


while solver.y > 1.01*y_max:
    t.append(solver.t)
    y.append(solver.y)
    try:
        solver.step()
    except:
        break
t_eq = solver.t

while solver.y < 0.99*y0:
    t.append(solver.t)
    y.append(solver.y)
    try:
        solver.step()
    except:
        break
t_org = solver.t

plt.semilogy(np.array(t)/3600.0, y)
plt.show()
"""
# contour plot stuff

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



"""
