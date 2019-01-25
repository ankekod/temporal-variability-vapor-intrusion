import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import LSODA
from scipy.interpolate import interp1d
import sqlite3
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

data_dir = './data/preferential-pathway-sensitivity/'
db_dir = '/home/jonathan/Dropbox/var/'
#db_dir = 'C:\\Users\\jstroem\\Dropbox\\var\\'

db = sqlite3.connect(db_dir + 'HillAFB.db')


data = pd.read_sql_query( "SELECT StopTime, Concentration AS IndoorConcentration FROM TDBasement;", db, )
data['StopTime'] = data['StopTime'].apply(pd.to_datetime)
data['IndoorConcentration'] = data['IndoorConcentration'].apply(np.log10).replace([-np.inf,np.inf], np.nan)
data.dropna(inplace=True)

phases = pd.read_sql_query("SELECT * from phases;", db)
phases['StartTime'] = phases['StartTime'] .apply(pd.to_datetime)
phases['StopTime'] = phases['StopTime'] .apply(pd.to_datetime)

phase1 = phases[(phases['CPM'] == 'off') & (phases['LandDrain'] == 'open')]
phase3 = phases[(phases['CPM'] == 'off') & (phases['LandDrain'] == 'closed')]

filter1 = (data['StopTime'] < phase1['StopTime'].values[0])
filter3 = (data['StopTime'] > phase3['StartTime'].values[0])

pp_status = lambda x: 'Open' if x < phase1['StopTime'].values[0] else ('Closed' if x > phase3['StartTime'].values[0] else 'CPM')
data['PP'] = data['StopTime'].apply(pp_status)



# diurnal analysis
#data.set_index('StopTime',inplace=True)
#rint(data)
data['Time'] = data['StopTime'].map(lambda x: x.strftime("%H:%M"))



data = data.loc[data['PP']=='Closed']
grp = data.groupby('Time').describe()
grp['Time'] = pd.to_datetime(data['Time'].astype(str))

grp['IndoorConcentration']['mean'].plot()

plt.show()
#ax = sns.lineplot( data=data)

#data.index = pd.to_datetime(data.index.astype(str))

"""
fig, ax = plt.subplots()
ax.plot(grp.index, grp['IndoorConcentration']['mean'], 'g', linewidth=2.0)
ax.plot(grp.index, grp['IndoorConcentration']['75%'], color='g')
ax.plot(grp.index, grp['IndoorConcentration']['25%'], color='g')
ax.fill_between(grp.index, grp['IndoorConcentration']['mean'], grp['IndoorConcentration']['75%'], alpha=.5, facecolor='g')
ax.fill_between(grp.index, grp['IndoorConcentration']['mean'], grp['IndoorConcentration']['25%'], alpha=.5, facecolor='g')


plt.show()
"""
