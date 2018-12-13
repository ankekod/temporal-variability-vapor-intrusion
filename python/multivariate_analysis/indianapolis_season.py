import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import LSODA
from scipy.interpolate import interp1d
import sqlite3
import seaborn as sns

data_dir = './data/preferential-pathway-sensitivity/'
db_dir = '/home/jonathan/Dropbox/var/'

db = sqlite3.connect(db_dir + 'Indianapolis.db')

observation = pd.read_sql_query(
    "SELECT StopDate AS StopTime FROM Observation_Status_Data WHERE Variable='Mitigation' AND Value='not yet installed';", db, )
observation['SSDStatus'] = np.repeat('SSD not installed', len(observation))

pressure = pd.read_sql_query(
    "SELECT StopDate AS StopTime, AVG(Value) AS Pressure FROM Differential_Pressure_Data WHERE Variable='Basement.Vs.Exterior' GROUP BY StopDate;", db, )

radon = pd.read_sql_query(
    "SELECT Date(StopDate) AS StopTime, AVG(Value) AS RadonConcentration FROM Radon_Data_SIRAD WHERE Variable='Radon' GROUP BY StopDate;", db, )

groundwater = pd.read_sql_query(
    "SELECT Date(StopDate) AS StopTime, AVG(Value) AS GroundwaterConcentration, Variable AS Specie FROM VOC_Data_Groundwater GROUP BY DATE(StopDate);", db, )

sub_surface = pd.read_sql_query(
    "SELECT StopDate AS StopTime, AVG(Value) AS SubSurfaceConcentration FROM VOC_Data_SoilGas_and_Air WHERE Depth_ft='Basement' GROUP BY StopDate;", db, )

weather = pd.read_sql_query(
    "SELECT StopDate AS StopTime, AVG(Value) AS Temperature FROM Meteorological_Data WHERE Variable='Temp.Out' AND Units='F' GROUP BY StopDate;", db, )

indoor = pd.read_sql_query(
    "SELECT Date(StopDate) AS StopTime, AVG(Value) AS IndoorConcentration, Variable AS Specie FROM VOC_Data_SRI_8610_Onsite_GC GROUP BY DATE(StopDate);", db, )

observation['StopTime'] = observation['StopTime'].apply(pd.to_datetime)
indianapolis = observation.set_index('StopTime')#.reset_index()
#print(indianapolis)

#rint(indianapolis['StopTime'])
i = 0
for df in (pressure,weather,groundwater,sub_surface,indoor):
    df['StopTime'] = df['StopTime'].apply(pd.to_datetime)
    try:
        indianapolis = indianapolis.combine_first(df.set_index('StopTime'))#.reset_index()
    except:
        print('%i failed' % i)
    for col in list(indianapolis):
        if col == 'index' or col == 'level_0':
            print(indianapolis[col])
            indianapolis.drop(columns=[col], inplace=True)
        else:
            continue
    i += 1

indianapolis = indianapolis.reset_index()
print(list(indianapolis))
#indianapolis['StopTime'] = indianapolis['StopTime'].apply(pd.to_datetime)

indianapolis['IndoorConcentration'] = indianapolis['IndoorConcentration'].apply(np.log10)

indianapolis.interpolate(inplace=True)
#df = df[df['Pressure'] < 35.0]
#df['Pressure'] *= -1
indianapolis.replace([np.inf, -np.inf], np.nan, inplace=True)


indianapolis = indianapolis[indianapolis['Specie'] == 'Trichloroethene']
indianapolis = indianapolis.drop(columns=['SSDStatus', 'Specie'])
indianapolis['Month'] = indianapolis['StopTime'].map(lambda x: x.month)
indianapolis['Temperature'] = indianapolis['Temperature'].map(lambda x: (x-32)/1.8)


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


indianapolis['Season'] = indianapolis['Month'].apply(lambda x: get_season(x))

indianapolis = indianapolis.drop(columns=['StopTime','Month'])

# TODO: change axis labels (adding units) and adjust ticks for the indoor concentration

sns.pairplot(indianapolis, hue="Season", hue_order=['Winter','Fall','Spring','Summer'])
plt.savefig('./figures/multivariate_analysis/indianapolis_season.pdf',dpi=300)
plt.show()
