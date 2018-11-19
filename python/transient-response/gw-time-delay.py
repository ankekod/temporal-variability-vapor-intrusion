import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import LSODA
from scipy.interpolate import interp1d
import sqlite3
import seaborn as sns
from scipy.stats import pearsonr

#db_dir = '/home/jonathan/lib/vapor-intrusion-dbs/'
db_dir = 'C://Users/jstroem/lib/vapor-intrusion-dbs/'

db = sqlite3.connect(db_dir + 'hill-afb.db')

asu = pd.read_sql_query(
    "SELECT\
        avgs.*, \
        gw.concentration as gw_concentration \
    FROM daily_averages avgs\
    LEFT JOIN average_groundwater_concentration gw ON (date(avgs.time) == date(gw.time))\
    UNION ALL\
    SELECT\
        avgs.*, \
        gw.concentration as gw_concentration \
    FROM average_groundwater_concentration gw\
    LEFT JOIN daily_averages avgs ON (date(avgs.time) == date(gw.time))\
    WHERE avgs.time IS NULL;",
    db,
).sort_values('time').reset_index().interpolate(method='piecewise_polynomial').dropna()
K_H = 0.403
phases = pd.read_sql_query("SELECT * from phases;", db)
phase3 = phases[(phases.cpm == 'off') & (phases.land_drain == 'closed')]




asu.time = asu.time.apply(pd.to_datetime)
filter3 = (asu['time'] > phase3['start'].values[0])

shifts = np.arange(0,len(asu)-1)
r = []
dt = []
var = []
fig, ax = plt.subplots()

for shift in shifts:
    df = asu.copy()
    df.gw_concentration = df.gw_concentration.shift(shift)
    df['alpha'] = df['concentration']/(df['gw_concentration']*1e3*K_H)
    df['alpha'] = df['alpha'].apply(np.log10)
    df = df.loc[filter3]
    df = df.dropna()
    r.append(pearsonr(df.concentration,df.gw_concentration)[0])
    var.append(df['alpha'].std())
    #df.plot(x='time',y=['concentration','gw_concentration'],logy=True,ax=ax,style=['-','--'],label=['Concentation, shift = %i' % shift, 'GW, shift = %i' % shift])
    dt.append(df.time.min() - asu.time.min())


time_delay = pd.DataFrame({'dt': dt, 'r': r, 'std': var})


time_delay.dt /= np.timedelta64('1','D')
time_delay.plot(y=['r','std'],ax=ax)

asu_shifted = asu.copy()
asu_shifted['gw_concentration'] = asu_shifted['gw_concentration'].shift(426)




asu['alpha'] = asu['concentration']/(asu['gw_concentration']*1e3*K_H)
asu_shifted['alpha'] = asu_shifted['concentration']/(asu_shifted['gw_concentration']*1e3*K_H)


fig, ax = plt.subplots()


asu = asu.loc[filter3]
asu_shifted = asu_shifted.loc[filter3]

asu.plot(x='time',y='alpha',logy='True',ax=ax,label='ASU')
asu_shifted.plot(x='time',y='alpha',logy='True',ax=ax,label='ASU Shifted')



fig, ax = plt.subplots()
asu['alpha'] = asu['alpha'].apply(np.log10)
asu_shifted['alpha'] = asu_shifted['alpha'].apply(np.log10)


sns.kdeplot(asu['alpha'],ax=ax,label='ASU')
sns.kdeplot(asu_shifted['alpha'],ax=ax,label='ASU Shifted')


plt.show()
