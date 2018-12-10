import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import LSODA
from scipy.interpolate import interp1d
import sqlite3
import seaborn as sns

data_dir = './data/preferential-pathway-sensitivity/'
#db_dir = '/home/jonathan/lib/vapor-intrusion-dbs/'
db_dir = '/home/jonathan/Dropbox/vapor-intrusion-dbs/'

db = sqlite3.connect(db_dir + 'hill-afb.db')

indoor = pd.read_sql_query( "SELECT * FROM DailyAverages;", db, )
soil_gas = pd.read_sql_query( "SELECT * FROM AverageSubSurfaceSoilGasConcentration;", db, )
groundwater = pd.read_sql_query( "SELECT StopTime, Concentration FROM AverageGroundWaterConcentration;", db, )

indoor.rename(columns={'Concentration': 'IndoorConcentration',}, inplace=True)
soil_gas.rename(columns={'Concentration': 'SubSurfaceConcentration',}, inplace=True)
groundwater.rename(columns={'Concentration': 'GroundwaterConcentration',}, inplace=True)


for df in (indoor,soil_gas,groundwater):
    df['StopTime'] = df['StopTime'].apply(pd.to_datetime)

df = indoor.set_index('StopTime').combine_first(soil_gas.set_index('StopTime')).reset_index().combine_first(groundwater.set_index('StopTime')).reset_index()

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


df[['IndoorConcentration','EmissionRate']] = df[['IndoorConcentration','EmissionRate']].apply(np.log10)
df.interpolate(inplace=True)
df = df[df['Pressure'] < 35.0]
df['Pressure'] *= -1
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.drop(columns=['index'],inplace=True)
df.dropna(inplace=True)


phases = pd.read_sql_query("SELECT * from phases;", db)
phases['StartTime'] = phases['StartTime'] .apply(pd.to_datetime)
phases['StopTime'] = phases['StopTime'] .apply(pd.to_datetime)

phase1 = phases[(phases['CPM'] == 'off') & (phases['LandDrain'] == 'open')]
phase3 = phases[(phases['CPM'] == 'off') & (phases['LandDrain'] == 'closed')]

filter1 = (df['StopTime'] < phase1['StopTime'].values[0])
filter3 = (df['StopTime'] > phase3['StartTime'].values[0])

pre_cpm = df.loc[filter1].copy()
post_cpm = df.loc[filter3].copy()


cols_to_drop = ['StopTime','Month','EmissionRate']
df.drop(columns=cols_to_drop,inplace=True)
pre_cpm.drop(columns=cols_to_drop,inplace=True)
post_cpm.drop(columns=cols_to_drop,inplace=True)
sns.pairplot(post_cpm, hue="Season", hue_order=['Winter','Fall','Spring','Summer'])

# TODO: change axis labels (adding units) and adjust ticks for the indoor concentration
plt.show()
