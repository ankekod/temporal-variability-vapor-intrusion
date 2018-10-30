import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import LSODA
from scipy.interpolate import interp1d
import sqlite3
import seaborn as sns

data_dir = './data/preferential-pathway-sensitivity/'
#db_dir = '/home/jonathan/lib/vapor-intrusion-dbs/'
db_dir = 'C://Users/jstroem/lib/vapor-intrusion-dbs/'

db_asu = sqlite3.connect(db_dir + 'hill-afb.db')
asu = pd.read_sql_query(
    "SELECT \
        time, \
        concentration \
    FROM \
        \"td-basement\";",
    db_asu,
)

"""
pressure = pd.read_sql_query(
    "SELECT \
        time, \
        pressure \
    FROM \
        pressure_difference;",
    db_asu,
).time.apply(pd.to_datetime)
"""

entry_rate = pd.read_sql_query(
    "SELECT \
        time, \
        emission_rate \
    FROM \
        daily_emission_rate;",
    db_asu,
)

tracer = pd.read_sql_query(
    "SELECT \
        time, \
        air_exchange_rate AS Ae \
    FROM \
        tracer;",
    db_asu,
)

phase3 = lambda df: df[(df.time - df.time.min())/np.timedelta64(1,'D') > 765.0].copy()

dfs = [asu, entry_rate, tracer]

for df in dfs:
    df.time = df.time.apply(pd.to_datetime)
    #df = phase3(df)


asu = phase3(asu)
entry_rate = phase3(entry_rate)
tracer = phase3(tracer)
"""
freqs = ['4H','12H','1D','3D','1W','1M']
fig, ax = plt.subplots()

for freq in freqs:
    resample = asu.resample(freq, on='time').mean().reset_index()
    resample['concentration'] = resample['concentration'].apply(np.log10)
    resample['dcdt'] = resample['concentration'].diff()#/(resample['time'].diff()/np.timedelta64(1,'D'))

    sns.kdeplot(
        data=resample['dcdt'].replace([np.inf, -np.inf], np.nan).dropna(),
        ax=ax,
    )
plt.show()

#print(asu.time.min())

"""


Ae = interp1d(tracer.time, tracer.Ae)
V = 350.0
t0 = entry_rate.time.min()#/np.timedelta64(1,'D')
print(t0)
tau = entry_rate.time.max()

y0 = entry_rate[entry_rate['time'] == t0]/V/Ae(t0)
print(y0)
def dudt(t, u):
    dudt =  n/V - u*Ae
    return dudt
solver = LSODA(
    dudt,
    t0,
    y0,
    tau,
    max_step=np.timedelta64(1,'D'),
)
