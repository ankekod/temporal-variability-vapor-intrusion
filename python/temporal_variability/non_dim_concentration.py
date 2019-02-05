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

pp_status = lambda x: 'Open' if x < phase1['StopTime'].values[0] else ('Closed' if x > phase3['StartTime'].values[0] else 'CPM')
asu['PP'] = asu['StopTime'].apply(pp_status)

asu_plot = asu.loc[(asu['PP']!='CPM') & (asu['Pressure'] > -5)]



for cond in asu_plot['PP'].unique():
    corr = asu_plot.loc[asu_plot['PP']==cond][['Pressure','AirExchangeRate','AttenuationSubSurface']].corr()
    corr['PP'] = np.repeat(cond, len(corr))
    corr['Variable'] = list(corr.index)
    try:
        corrs = corrs.append(corr, ignore_index=True)
    except:
        corrs = corr
corrs.set_index('Variable', inplace=True)
#corrs.to_csv('./pearson.csv')


g = sns.PairGrid(asu_plot[['Pressure','AirExchangeRate','AttenuationSubSurface','PP']], hue='PP')
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.regplot, x_bins=20, truncate=True, )
g = g.map_diag(sns.kdeplot, shade=True)
g = g.add_legend()

# non-linear fit

def func(p, a, b):
    Ae = a*p + b
    return Ae

xdata = asu.loc[(asu['PP']!='CPM') & (asu['Pressure'] > -5)]['Pressure'].values
ydata = asu.loc[(asu['PP']!='CPM') & (asu['Pressure'] > -5)]['AirExchangeRate'].values

popt, pcov = curve_fit(func, xdata, ydata)


fig, ax = plt.subplots()
plt.plot(xdata,ydata,'o',label='Data')
#plt.plot(xdata, func(xdata, *popt), 'o', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
xdata= np.linspace(-15,15)
plt.plot(xdata, func(xdata, *popt), 'o', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))


plt.legend()


dfs = []
files = (
    'no_pp_gravel_sub-base',
    'no_pp_uniform_soil',
    'pp_gravel_sub-base',
    'pp_uniform_soil',
)
for file in files:
    sim = pd.read_csv('./data/preferential-pathway-sensitivity/%s.csv' % file, header=4)
    sim['Simulation'] = np.repeat(file, len(sim))
    dfs.append(sim)

sim = pd.concat(dfs)


sim.rename(columns={
        '% p_in': 'Pressure',
        'Attenuation factor, Global Evaluation: Attenuation factor {gev2}': 'AttenuationGroundwater',
        'Global Evaluation: Relative air entry rate {gev3}': 'RelativeEntryRate',
        'Average crack Peclet number, Global Evaluation: Crack Peclet number {gev4}': 'Pe',
        'TCE in indoor air, Global Evaluation: TCE in indoor air {gev5}': 'IndoorConcentration',
        'Global Evaluation: TCE emission rate {gev6}': 'EmissionRate',
    },
    inplace=True,
)

fig, ax = plt.subplots()


data = asu.loc[(asu['PP'] != 'CPM') & (asu['Pressure'] >= -5.5)]

for hue in data['PP'].unique():

    sns.regplot(x='Pressure', y='AttenuationGroundwater', data=data.loc[data['PP']==hue], ax=ax, fit_reg = False, x_bins=np.linspace(-6,6,40))

sns.lineplot(x='Pressure',y='AttenuationGroundwater',hue='Simulation',data=sim, ax=ax)
ax.set_yscale('log')

plt.show()
