import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import LSODA
from scipy.interpolate import interp1d
import sqlite3
import seaborn as sns
from scipy import stats


data_dir = './data/preferential-pathway-sensitivity/'
db_dir = '/home/jonathan/Dropbox/var/'
#db_dir = 'C:\\Users\\jstroem\\Dropbox\\var\\'

db = sqlite3.connect(db_dir + 'HillAFB.db')

pressure = pd.read_sql_query( "SELECT StopTime, AVG(Pressure) AS Pressure FROM PressureDifference GROUP BY DATE(StopTime);", db, )


indoor = pd.read_sql_query( "SELECT StopTime, AVG(Concentration) AS IndoorConcentration FROM TDBasement GROUP BY DATE(StopTime);", db, )
air_exchange_rate = pd.read_sql_query( "SELECT StopTime, AVG(AirExchangeRate) AS AirExchangeRate FROM Tracer GROUP BY DATE(StopTime);", db, )
soil_gas = pd.read_sql_query( "SELECT StopTime, MAX(Concentration) AS SubSurfaceConcentration FROM SoilGasConcentration WHERE Depth=0.0 AND (Location='1' OR Location='2' OR Location='3' OR Location='4' OR Location='5' OR Location='6' OR Location='7') GROUP BY StopTime;", db, )
groundwater = pd.read_sql_query( "SELECT StopTime, AVG(Concentration) AS GroundwaterConcentration FROM GroundwaterConcentration WHERE Depth=2.7 GROUP BY StopTime;", db, )



for asu in (indoor,soil_gas,groundwater):
    asu['StopTime'] = asu['StopTime'].apply(pd.to_datetime)

asu = pressure.set_index('StopTime').combine_first(indoor.set_index('StopTime')).combine_first(air_exchange_rate.set_index('StopTime')).combine_first(soil_gas.set_index('StopTime')).combine_first(groundwater.set_index('StopTime')).reset_index()
asu.interpolate(inplace=True,type='linear')
asu.dropna(inplace=True)


asu['Month'] = asu['StopTime'].map(lambda x: x.month)


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
asu['Season'] = asu['Month'].apply(lambda x: get_season(x))
asu['AttenuationSubSurface'] = asu['IndoorConcentration']/asu['SubSurfaceConcentration']
asu['AttenuationSubSurface'] = asu['AttenuationSubSurface'].apply(np.log10).replace([-np.inf,np.inf], np.nan)
asu.dropna(inplace=True)
asu['Pressure'] *= -1
#asu = asu[asu['Pressure']>-5]

K_H = 0.403 # Henry's Law constant
asu['AttenuationGroundwater'] = asu['IndoorConcentration']/asu['GroundwaterConcentration']/1e3/K_H

phases = pd.read_sql_query("SELECT * from phases;", db)
phases['StartTime'] = phases['StartTime'] .apply(pd.to_datetime)
phases['StopTime'] = phases['StopTime'] .apply(pd.to_datetime)

phase1 = phases[(phases['CPM'] == 'off') & (phases['LandDrain'] == 'open')]
phase3 = phases[(phases['CPM'] == 'off') & (phases['LandDrain'] == 'closed')]

filter1 = (asu['StopTime'] < phase1['StopTime'].values[0])
filter3 = (asu['StopTime'] > phase3['StartTime'].values[0])

pre_cpm = asu.loc[filter1].copy()
post_cpm = asu.loc[filter3].copy()

# Attenunation from subsurface concentration correlation with pressure and air exchange plots
fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,sharey=True)

r, p = stats.pearsonr( pre_cpm['Pressure'], pre_cpm['AttenuationSubSurface'],)
ax1.set_title('r = %1.2f' % r)

r, p = stats.pearsonr( pre_cpm['AirExchangeRate'], pre_cpm['AttenuationSubSurface'],)
ax2.set_title('r = %1.2f' % r)
r, p = stats.pearsonr( post_cpm['Pressure'], post_cpm['AttenuationSubSurface'],)

ax3.set_title('r = %1.2f' % r)
r, p = stats.pearsonr( post_cpm['AirExchangeRate'], post_cpm['AttenuationSubSurface'],)

ax4.set_title('r = %1.2f' % r)

sns.kdeplot(pre_cpm['Pressure'], pre_cpm['AttenuationSubSurface'],clip=((-3,3),(-4,0)), shade_lowest=False,shade=True,  ax=ax1)
sns.kdeplot(pre_cpm['AirExchangeRate'], pre_cpm['AttenuationSubSurface'],clip=((0,1),(-4,0)), shade_lowest=False,shade=True,  ax=ax2)

sns.kdeplot(post_cpm['Pressure'], post_cpm['AttenuationSubSurface'],clip=((-3,3),(-4,0)), shade_lowest=False,shade=True,  ax=ax3)
sns.kdeplot(post_cpm['AirExchangeRate'], post_cpm['AttenuationSubSurface'],clip=((0,1),(-4,0)), shade_lowest=False,shade=True,  ax=ax4)


plt.tight_layout()
plt.show()
