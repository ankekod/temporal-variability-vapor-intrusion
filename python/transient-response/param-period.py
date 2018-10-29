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

#asu['concentration'] = asu['concentration'].apply(np.log10)

asu['dcdt'] = asu['concentration'].diff()/(asu['time'].diff()/np.timedelta64(1,'h'))

fig, ax = plt.subplots()

asu.plot(
    x='time',
    y='concentration',
    #ax=ax1,
    logy=True,
    style='o',
    ax=ax,
)
entry_rate.plot(
    x='time',
    y='emission_rate',
    logy=True,
    style='o',
    #color='orange',
    ax=ax,
)

#print(asu.time.min())

"""
def dudt(t, u):
    dudt =  n/V - u*Ae
    return dudt
solver = LSODA(
    dudt,
    t0,
    y0.values,
    tau,
    max_step=1.0,
)
"""
plt.show()
