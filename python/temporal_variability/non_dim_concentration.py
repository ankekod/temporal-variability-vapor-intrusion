import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import LSODA
from scipy.interpolate import interp1d
import sqlite3
import seaborn as sns

data_dir = './data/preferential-pathway-sensitivity/'
db_dir = '/home/jonathan/Dropbox/var/'
db_dir = 'C:\\Users\\jstroem\\Dropbox\\var\\'

db = sqlite3.connect(db_dir + 'HillAFB.db')

pressure = pd.read_sql_query( "SELECT StopTime, Pressure FROM PressureDifference;", db, )
indoor = pd.read_sql_query( "SELECT StopTime, Concentration AS IndoorConcentration FROM TDBasement;", db, )
air_exchange_rate = pd.read_sql_query( "SELECT StopTime, AirExchangeRate FROM Tracer;", db, )
soil_gas = pd.read_sql_query( "SELECT StopTime, MAX(Concentration) AS SubSurfaceConcentration FROM SoilGasConcentration WHERE Depth=0.0 AND (Location='1' OR Location='2' OR Location='3' OR Location='4' OR Location='5' OR Location='6' OR Location='7') GROUP BY StopTime;", db, )
groundwater = pd.read_sql_query( "SELECT StopTime, AVG(Concentration) AS GroundwaterConcentration FROM GroundwaterConcentration WHERE Depth=2.7 GROUP BY StopTime;", db, )



for df in (indoor,soil_gas,groundwater):
    df['StopTime'] = df['StopTime'].apply(pd.to_datetime)

df = pressure.set_index('StopTime').combine_first(indoor.set_index('StopTime')).combine_first(air_exchange_rate.set_index('StopTime')).combine_first(soil_gas.set_index('StopTime')).combine_first(groundwater.set_index('StopTime')).reset_index()
df.interpolate(inplace=True,type='linear')
df.dropna(inplace=True)


df['Month'] = df['StopTime'].map(lambda x: x.month)


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
df['Season'] = df['Month'].apply(lambda x: get_season(x))
df['AttenuationSubSurface'] = df['IndoorConcentration']/df['SubSurfaceConcentration']
K_H = 0.403 # Henry's Law constant
df['AttenuationGroundwater'] = df['IndoorConcentration']/df['GroundwaterConcentration']/1e3/K_H

r = df.resample('1D' ,on='StopTime', kind='timestamp').mean()

phases = pd.read_sql_query("SELECT * from phases;", db)
phases['StartTime'] = phases['StartTime'] .apply(pd.to_datetime)
phases['StopTime'] = phases['StopTime'] .apply(pd.to_datetime)

phase1 = phases[(phases['CPM'] == 'off') & (phases['LandDrain'] == 'open')]
phase3 = phases[(phases['CPM'] == 'off') & (phases['LandDrain'] == 'closed')]

filter1 = (df['StopTime'] < phase1['StopTime'].values[0])
filter3 = (df['StopTime'] > phase3['StartTime'].values[0])

pre_cpm = df.loc[filter1].copy()
post_cpm = df.loc[filter3].copy()


sns.pairplot(r[['AttenuationSubSurface','Pressure','AirExchangeRate','Season']], hue="Season", hue_order=['Winter','Fall','Spring','Summer'])


plt.show()
