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

asu.time = asu.time.apply(pd.to_datetime)

shifts = np.arange(0,len(asu)-1)
r = []
dt = []
fig, ax = plt.subplots()

for shift in shifts:
    df = asu.copy()
    df.gw_concentration = df.gw_concentration.shift(shift)
    df = df.dropna()
    r.append(pearsonr(df.gw_concentration,df.concentration)[0])
    #df.plot(x='time',y=['concentration','gw_concentration'],logy=True,ax=ax,style=['-','--'],label=['Concentation, shift = %i' % shift, 'GW, shift = %i' % shift])
    dt.append(df.time.min() - asu.time.min())


time_delay = pd.DataFrame({'dt': dt, 'r': r})


time_delay.dt /= np.timedelta64('1','D')


time_delay.plot(x='dt',y='r',ax=ax)
plt.show()
