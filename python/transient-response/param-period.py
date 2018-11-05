import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import LSODA
from scipy.interpolate import interp1d
import sqlite3
import seaborn as sns

data_dir = './data/preferential-pathway-sensitivity/'
db_dir = '/home/jonathan/lib/vapor-intrusion-dbs/'
#db_dir = 'C://Users/jstroem/lib/vapor-intrusion-dbs/'


db = sqlite3.connect(db_dir + 'hill-afb.db')

asu = pd.read_sql_query("SELECT * from daily_averages;", db)
phases = pd.read_sql_query("SELECT * from phases;", db)
asu.Ae *= 24.0
asu.t = asu.t.apply(pd.to_datetime)
phases.start = phases.start.apply(pd.to_datetime)
phase = phases[(phases.cpm == 'off') & (phases.land_drain == 'closed')]
filter = (asu['t'] > phase['start'].values[0])
asu = asu.loc[filter]

asu.t = (asu.t - asu.t.min())/np.timedelta64(1,'D')
asu.t = asu.t.apply(float)

#asu['c'] = asu['c'].apply(np.log10)
asu['dcdt'] = asu['c'].diff()/asu['t'].diff()


print(asu['dcdt'].mean())
