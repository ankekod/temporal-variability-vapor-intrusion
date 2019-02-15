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
#db_dir = '/home/jonathan/Dropbox/var/'
db_dir = 'C:\\Users\\jstroem\\Dropbox\\var\\'

db = sqlite3.connect(db_dir + 'HillAFB.db')

pressure = pd.read_sql_query( "SELECT StopTime, Pressure FROM PressureDifference;", db, )
pressure['StopTime'] = pressure['StopTime'].apply(pd.to_datetime)
pressure['Pressure'] *= -1

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
pressure['PP'] = pressure['StopTime'].apply(pp_status)

def get_season(x):
    seasons = {
        'Winter': (12, 2),
        'Spring': (3, 5),
        'Summer': (6, 8),
        'Fall': (9, 11),
    }
    if (x == 12) or (x == 1) or (x == 2):
        return 'Winter'
    elif (x == 3) or (x == 4) or (x == 5):
        return 'Spring'
    elif (x == 6) or (x == 7) or (x == 8):
        return 'Summer'
    elif (x == 9) or (x == 10) or (x == 11):
        return 'Fall'
    else:
        return 'Error'


# finds the month and season of each timestamp
pressure['Month'] = pressure['StopTime'].map(lambda x: x.month)
pressure['Season'] = pressure['Month'].apply(lambda x: get_season(x))
#pressure = pressure.loc[pressure['PP']=='Closed']
#r = pressure.resample('1H' ,on='StopTime', kind='timestamp')
#r = r['Pressure'].agg([np.mean, np.max, np.min, np.std])

fig, ax = plt.subplots()

for status in ('Open','Closed',):
    for season in pressure['Season'].unique():
        print('PP status: ' + status + ', Season: ' + season)
        df = pressure.loc[(pressure['PP']==status) & (pressure['Season']==season)]
        r = df[['StopTime','Pressure']].groupby(pressure['StopTime'].dt.hour).mean()
        ax.plot(r.index, r['Pressure'].values, label = "PP status: %s, Season: %s" % (status, season))
        print(r)
        r.to_csv('./data/diurnal/pressure_%s_%s.csv' % (status.lower(), season.lower()))

ax.legend(loc='best')
plt.show()
# diurnal analysis
#data.set_index('StopTime',inplace=True)
#rint(data)
data['Time'] = data['StopTime'].map(lambda x: x.strftime("%H:%M"))
#pressure['Time'] = pressure['StopTime'].map(lambda x: x.strftime("%H:%M"))


"""
data = data.loc[data['PP']=='Open']
grp = data.groupby('Time').describe()
grp['Time'] = pd.to_datetime(data['Time'].astype(str))
grp['IndoorConcentration']['mean'].plot()
"""


"""
pressure = pressure.loc[pressure['PP']=='Closed']
grp = pressure.groupby('Time').describe()
grp['Time'] = pd.to_datetime(pressure['Time'].astype(str))
#grp.set_index('Time', inplace=True)
grp['Pressure']['mean'].plot()
"""


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
