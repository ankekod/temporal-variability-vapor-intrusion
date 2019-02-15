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
    "SELECT\
        time, pressure\
    FROM \
        pressure_difference;",
    db,
).sort_values('time').dropna()



# total pressure overiew
fig, ax = plt.subplots()
asu.plot(x='time',y='pressure',ax=ax)

# distribution of pressure
fig, ax = plt.subplots()
sns.kdeplot(asu.dropna()['pressure'],gridsize=3e3,ax=ax)


asu['dp'] = asu['pressure'].diff()
asu.time = asu.time.apply(pd.to_datetime)
phases = pd.read_sql_query("SELECT * from phases;", db)
asu['month'] = asu['time'].map(lambda x: x.month) # assign integer for each month

seasons = {
    'winter': (12,2),
    'spring': (3,5),
    'summer': (6,8),
    'fall': (9,11),
}

def get_season(x):
    if (x==12) or (x==1) or (x==2):
        return 'winter'
    elif (x==3) or (x==4) or (x==5):
        return 'spring'
    elif (x==6) or (x==7) or (x==8):
        return 'summer'
    elif (x==9) or (x==10) or (x==11):
        return 'fall'
    else:
        return 'error'
asu['season'] = asu['month'].apply(lambda x: get_season(x))

phases.start = phases.start.apply(pd.to_datetime)
phase1 = phases[(phases.cpm == 'off') & (phases.land_drain == 'open')]
phase3 = phases[(phases.cpm == 'off') & (phases.land_drain == 'closed')]
filter1 = (asu['time'] < phase1['stop'].values[0])
filter3 = (asu['time'] > phase3['start'].values[0])

pre_cpm = asu.loc[filter1].copy()
post_cpm = asu.loc[filter3].copy()

convert_time = lambda df: (df.time.diff()/np.timedelta64(1,'m')).apply(float)

pre_cpm['dt'] = convert_time(pre_cpm)
post_cpm['dt'] = convert_time(post_cpm)

pre_cpm['dpdt'] = pre_cpm['dp']/pre_cpm['dt']
post_cpm['dpdt'] = post_cpm['dp']/post_cpm['dt']

# dpdt distribution plots

fig, ax = plt.subplots()

sns.kdeplot(pre_cpm.dropna()['dpdt'],gridsize=3e3,ax=ax,label='PP open')
sns.kdeplot(post_cpm.dropna()['dpdt'],gridsize=3e3,ax=ax,label='PP closed')
plt.legend()
plt.show()
