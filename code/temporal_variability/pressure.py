import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy import stats
import sqlite3
db_dir = 'C:\\Users\\jstroem\\Dropbox\\var\\'

db = sqlite3.connect(db_dir + 'HillAFB.db')

asu = pd.read_sql_query("SELECT StopTime AS Time, Pressure FROM PressureDifference;",
        db,
    ).sort_values(by='Time')
asu['Time'] = asu['Time'].apply(pd.to_datetime)


phases = pd.read_sql_query("SELECT * from phases;", db)
phases['StartTime'] = phases['StartTime'] .apply(pd.to_datetime)
phases['StopTime'] = phases['StopTime'] .apply(pd.to_datetime)

phase1 = phases[(phases['CPM'] == 'off') & (phases['LandDrain'] == 'open')]
phase3 = phases[(phases['CPM'] == 'off') & (phases['LandDrain'] == 'closed')]

pp_status = lambda x: 'Open' if x < phase1['StopTime'].values[0] else ('Closed' if x > phase3['StartTime'].values[0] else 'CPM')
asu['Phase'] = asu['Time'].apply(pp_status)

asu = asu.loc[asu['Phase']!='CPM']

asu['dp'] = asu['Pressure'].diff().apply(np.abs)
asu['dt'] = asu['Time'].diff()/np.timedelta64(1, 'm')
asu['dpdt'] = asu['dp']/asu['dt']
asu['dpdt'] = asu['dpdt'].replace([-np.inf,np.inf], np.nan)
asu.dropna(inplace=True)

print(asu['dpdt'].describe(percentiles=[0.01, 0.05, 0.95, 0.99]))


sns.boxplot(
    x="Phase",
    y="dpdt",
    data=asu,
)
plt.show()
