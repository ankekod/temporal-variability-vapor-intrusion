import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import seaborn as sns

data_dir = './data/preferential-pathway-sensitivity/'
#db_dir = '/home/jonathan/lib/vapor-intrusion-dbs/'
db_dir = 'C://Users/jstroem/lib/vapor-intrusion-dbs/'

"""
#Ae = 12.0 # 1/day
M = 131.38 # TCE g/mol
V = 300.0 # basement volume m3

def Ae(p):
    return 12.0*p

def dp(t, period):
    dp = 10.0*np.sin(2.0*np.pi*t/period) - 0.5
    return dp

df = pd.read_csv('./data/transient-response/param-period2.csv', header=4)
df['Pressure'] = dp(df['Time'], df['period'])
df['TCE expulsion rate'] = df['TCE in indoor air']*Ae(df['Pressure'])*M*V


df2 = pd.read_csv('data/preferential-pathway-sensitivity/param-preferential-pathway.csv', header=4)
reference_lines = df2[((df2['chi'] == 1) & (df2['SB']==1)) & ((df2['p'] == 10.5) | (df2['p'] == -10.5))]


pivoted_df = df.pivot(
    index='Time',
    columns='period',
    values=['Attenuation factor','Pressure','TCE emission rate','TCE expulsion rate'],
).interpolate(
    method='piecewise_polynomial',
)

for period in df['period'].unique():
    print(df['Pressure'][df['period']==period].median())

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

pivoted_df.plot(
    logy=True,
    logx=True,
    y=['Attenuation factor'],
    #secondary_y='Pressure',
    ax=ax1,
)

pivoted_df.plot(
    #logy=True,
    logx=True,
    y=['Pressure'],
    #secondary_y='Pressure',
    linestyle='--',
    ax=ax2,
)

t0 = 0
tau = pivoted_df.index.values[-1]


ax1.loglog(
    [t0, tau],
    [df2['alpha'].min(), df2['alpha'].min()],
    'k--',
)

ax1.loglog(
    [t0, tau],
    [df2['alpha'].max(), df2['alpha'].max()],
    'k--',
)

plt.show()

# check what the average p is over each period, I think it should be higher for
# the larger sin waves? could be interesting to see if you essentially achieve
# the ss alpha value that corresponds to this average pressure value ?

"""

db_asu = sqlite3.connect(db_dir + 'asu_house.db')
asu = pd.read_sql_query(
    "SELECT \
        day AS Day, \
        pressure_difference AS Pressure, \
        tce_emission_rate, \
        building_flow_rate, tce_groundwater \
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


#asu['Attenuation factor'] = asu['Attenuation factor'].apply(np.log10)
asu['Day'] *= 24.0*3600.0


asu['dcdt'] = pd.Series(asu['Attenuation factor'].diff()/asu['Day'].diff())
asu['dcdt'] = asu['dcdt'].apply(np.log10)

asu['dcdt'] = asu['dcdt'].apply(np.abs)

"""
asu[asu['PP'] == 'Open'][['Day','Attenuation factor']].apply(np.diff).apply(np.abs).plot(
    x='Day',
    y='Attenuation factor',
    logy=True,
    kind='scatter',
)

asu.plot(
    x='Day',
    y='dcdt',
    logy=True,
    kind='scatter',
)
"""
fig, ax = plt.subplots()

sns.kdeplot(
    asu['dcdt'],
    ax=ax,
)

#ax.set(xscale="log")


def dp(t, period):
    dp = 10.0*np.sin(2.0*np.pi*t/period) - 0.5
    return dp

df = pd.read_csv('./data/transient-response/param-period2.csv', header=4)
df['Pressure'] = dp(df['Time'], df['period'])


df['dcdt'] = pd.Series(df['Attenuation factor'].diff()/df['Time'].diff())
df['dcdt'] = df['dcdt'].apply(np.log10)

df['dcdt'] = df['dcdt'].apply(np.abs)

pivoted_df = df.pivot(
    index='Time',
    columns='period',
    values=['dcdt'],
).interpolate(
    method='piecewise_polynomial',
)


pivoted_df.plot(
    y='dcdt',
    kind='kde',
    ax=ax,
)

"""
sns.kdeplot(
    pivoted_df['dcdt'],
    ax=ax,
)
"""
ax.set_ylim([0,2])

plt.show()
