import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import seaborn as sns

data_dir = './data/preferential-pathway-sensitivity/'
#db_dir = '/home/jonathan/lib/vapor-intrusion-dbs/'
db_dir = 'C://Users/jstroem/lib/vapor-intrusion-dbs/'

db_asu = sqlite3.connect(db_dir + 'asu_house.db')
asu = pd.read_sql_query(
    "SELECT \
        day AS Day, \
        pressure_difference AS Pressure, \
        tce_emission_rate, \
        building_flow_rate, \
        tce_groundwater, \
        exchange_rate AS Ae \
    FROM \
        parameters; \
    ", db_asu,
).interpolate(limit_area='inside',limit=250).dropna()
# removes data that is outside model data range
asu['Pressure'] = -1.0*asu['Pressure']
asu = asu[(asu['Pressure'] < 10) & (asu['Pressure'] > -10)]
# removes cpm period from dataset
cpm_start, cpm_end = 780.0, 1157.0
asu = asu[(asu['Day'] < cpm_start) | (asu['Day'] > cpm_end)]
# assigns a PP open/closed column (for sorting purposes)
asu['PP'] = pd.Series(asu['Day'].apply(lambda x: 'Open' if x < cpm_start else 'Closed'))
# calculates attentuation factor
asu['Concentration'] = asu['tce_emission_rate']/asu['building_flow_rate']
asu['Attenuation factor'] = asu['Concentration']/asu['tce_groundwater']
asu['Ae'] *= 3600.0
"""
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

asu[asu.PP == 'Closed'].plot(
    x = 'Day',
    y = 'Attenuation factor',
    ax=ax1,
    logy=True,
)

asu[asu.PP == 'Closed'].plot(
    x = 'Day',
    y = 'Ae',
    color='orange',
    ax=ax2,
)
"""
fig, axarr = plt.subplots(2,2, sharey=True)

i = 0
for state in asu.PP.unique():
    for var in ('Ae', 'Pressure'):
        ax = axarr.flatten()[i]
        sns.kdeplot(
            data=asu[asu.PP==state][var],
            data2=asu[asu.PP==state]['Attenuation factor'].apply(np.log10),
            ax=ax,
        )
        ax.set_title('%s, r= %1.1f' % (state, corr))
        i += 1

plt.show()
