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


db = sqlite3.connect(db_dir + 'hill-afb.db')



asu = pd.read_sql_query(
    "SELECT \
        daily_averages.*, \
        AVG(groundwater_concentration.concentration) as gw_concentration \
    FROM \
        daily_averages \
    LEFT JOIN \
        groundwater_concentration \
    ON \
        date(daily_averages.time) = date(groundwater_concentration.time) \
    GROUP BY \
        date(daily_averages.time) \
    ;",
    db,
).interpolate(method='piecewise_polynomial')

asu.time = asu.time.apply(pd.to_datetime)

"""
gw = pd.read_sql_query(
    "SELECT \
        time, \
        AVG(concentration) as gw_conc \
    FROM \
        groundwater_concentration\
    GROUP BY \
        date(time);",
    db,
).interpolate(method='piecewise_polynomial')
"""
K_H = 0.403
asu['alpha'] = asu['concentration']/(asu['gw_concentration']*1e3*K_H)


phases = pd.read_sql_query("SELECT * from phases;", db)
asu.air_exchange_rate *= 24.0
asu.time = asu.time.apply(pd.to_datetime)
phases.start = phases.start.apply(pd.to_datetime)
phase = phases[(phases.cpm == 'off') & (phases.land_drain == 'closed')]
filter = (asu['time'] > phase['start'].values[0])
asu = asu.loc[filter]


asu.time = (asu.time - asu.time.min())/np.timedelta64(1,'D')
asu.time = asu.time.apply(float)



asu.plot(x='time',y=['concentration','alpha'],logy=True)
plt.show()

#asu['c'] = asu['c'].apply(np.log10)
#asu['dcdt'] = asu['c'].diff()/asu['t'].diff()
