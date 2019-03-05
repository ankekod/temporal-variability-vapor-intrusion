import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import LSODA
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import sqlite3
import seaborn as sns


data_dir = './data/preferential-pathway-sensitivity/'
db_dir = '/home/jonathan/Dropbox/var/'
fig_dir = './figures/rate_of_change/'
db = sqlite3.connect(db_dir + 'HillAFB.db')
db_indianapolis = sqlite3.connect(db_dir + 'Indianapolis.db')

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

class ProcessData:
    def __init__(self,db):

        return

    def get_rate_of_change(self, df, time_scale='1h'):
        df['TimeChange'] = (df['StopTime'].diff()/np.timedelta64(time_scale[0], time_scale[1])).apply(float)
        df['RateOfChange'] = df['Concentration'].diff()/df['TimeChange']
        return df

    def get_season(self, df):
        # finds the month and season of each timestamp
        asu['Month'] = asu['StopTime'].map(lambda x: x.month)
        asu['Season'] = asu['Month'].apply(lambda x: get_season(x))
        return df

def process_asu(resample_time=False):
    asu = pd.read_sql_query("SELECT * FROM TDBasement;",
        db,
    ).sort_values(by='StopTime')

    asu['StopTime'] = asu['StopTime'].apply(pd.to_datetime)

    if resample_time != False:
        asu = asu.resample(resample_time, on='StopTime', kind='timestamp').mean().reset_index()

    asu['Concentration'] = asu['Concentration'].apply(np.log10).replace([np.inf,-np.inf],np.nan)
    asu = asu.dropna().reset_index(drop=True)

    def rate_of_change(df, time_scale='1h'):
        df['TimeChange'] = (df['StopTime'].diff()/np.timedelta64(time_scale[0], time_scale[1])).apply(float)
        df['RateOfChange'] = df['Concentration'].diff()/df['TimeChange']
        return df

    if resample_time != False:
        asu = rate_of_change(asu,time_scale=resample_time)
        asu = asu.dropna().reset_index(drop=True)
    else:
        asu = rate_of_change(asu)
        asu = asu.dropna().reset_index(drop=True)

    # finds the month and season of each timestamp
    asu['Month'] = asu['StopTime'].map(lambda x: x.month)
    asu['Season'] = asu['Month'].apply(lambda x: get_season(x))
    # loads the different phases at ASU
    phases = pd.read_sql_query("SELECT * from Phases;", db)
    # converts time strings to datetime
    phases['StartTime'] = phases['StartTime'].apply(pd.to_datetime)
    phases['StopTime'] = phases['StopTime'].apply(pd.to_datetime)
    # identifies the phases phase 1 = pre_cpm and phase 2 = post_cpm
    phase1 = phases[(phases['CPM'] == 'off') & (phases['LandDrain'] == 'open')]
    phase3 = phases[(phases['CPM'] == 'off') & (phases['LandDrain'] == 'closed')]
    # applies phase filters
    filter1 = (asu['StopTime'] < phase1['StopTime'].values[0])
    filter3 = (asu['StopTime'] > phase3['StartTime'].values[0])
    pre_cpm = asu.loc[filter1].copy()
    post_cpm = asu.loc[filter3].copy()

    return asu, pre_cpm, post_cpm

def process_indianapolis(resample_time=False,specie='Tetrachloroethene'):
    indianapolis = pd.read_sql_query(
        "SELECT C.StopDate, C.StopTime AS TimeStamp, C.Value AS Concentration FROM VOC_Data_SRI_8610_Onsite_GC C, Observation_Status_Data O WHERE DATE(C.StopDate)=DATE(O.StopDate) AND O.Value='not yet installed' AND C.Variable=\'%s\';" % specie, db_indianapolis, )

    indianapolis = indianapolis.assign(StopTime=lambda x: x['StopDate']+' '+x['TimeStamp'])
    indianapolis.drop(columns=['StopDate','TimeStamp'],inplace=True)
    indianapolis['StopTime'] = indianapolis['StopTime'].apply(pd.to_datetime)

    if resample_time != False:
        indianapolis = indianapolis.resample(resample_time, on='StopTime', kind='timestamp').mean().reset_index()

    indianapolis['Concentration'] = indianapolis['Concentration'].apply(np.log10).replace([np.inf,-np.inf],np.nan)
    indianapolis = asu.dropna().reset_index(drop=True)

    def rate_of_change(df, time_scale='1h'):
        df['TimeChange'] = (df['StopTime'].diff()/np.timedelta64(time_scale[0], time_scale[1])).apply(float)
        df['RateOfChange'] = df['Concentration'].diff()/df['TimeChange']
        return df

    if resample_time != False:
        indianapolis = rate_of_change(indianapolis,time_scale=resample_time)
    else:
        indianapolis = rate_of_change(indianapolis)

    # finds the month and season of each timestamp
    indianapolis['Month'] = indianapolis['StopTime'].map(lambda x: x.month)
    indianapolis['Season'] = indianapolis['Month'].apply(lambda x: get_season(x))

    return indianapolis


def process_simulation(path):
    simulation = pd.read_csv(path)
    mol_m3_to_ug_m3 = 131.4 * 1e6
    simulation[['Cmax', 'Cmin']] *= mol_m3_to_ug_m3
    simulation[['Cmax', 'Cmin']] = simulation[['Cmax', 'Cmin']].apply(np.log10)
    #simulation[['TimeDown','TimeUp']] /= 24.0
    simulation['RateOfChangeUp'] = (simulation['Cmax'] - simulation['Cmin']) / simulation['TimeUp']
    simulation['RateOfChangeDown'] = (simulation['Cmin'] - simulation['Cmax']) / simulation['TimeDown']
    simulation = simulation.sort_values(by='RateOfChangeUp',ascending=False).reset_index(drop=True)

    return simulation

def custom_x_ticks(ax, my_xticks=np.log10([0.5, 1.0, 5.0])):
    my_xtick_labels = ["%.2f" % x_tick for x_tick in 10.0**my_xticks]
    ax.set_xlim((my_xticks[0], my_xticks[-1]))
    ax.set_xticks(my_xticks)
    ax.set_xticklabels(my_xtick_labels)
    ax.set_xlabel('$\\frac{\\Delta \\log(c)}{\\Delta \\mathrm{hour}}$')
    ax.set_ylabel('Probability density')
    plt.tight_layout()
    return


asu, pre_cpm, post_cpm = process_asu()
simulation = process_simulation('./data/transient-response/cstr-changes.csv')
simulation_pp = process_simulation('./data/transient-response/cstr-changes-pp.csv')


indianapolis = process_indianapolis()
#indianapolis.to_csv('./data/rate_of_change/indianapolis.csv',index=False)
#indianapolis = pd.read_csv('./data/rate_of_change/indianapolis.csv')

# figure 1: differences in rate of change for pre and post CPM periods
# and simulated comparison
fig, ax = plt.subplots()
sns.kdeplot(pre_cpm['RateOfChange'], gridsize=5e3, ax=ax, label='ASU, PP open')
sns.kdeplot(post_cpm['RateOfChange'], gridsize=5e3, ax=ax, label='ASU, PP closed')
sns.kdeplot(indianapolis['RateOfChange'], gridsize=1e4, ax=ax, label='Indianapolis, PCE')

i = 0
labels = ('Predicted, no PP', 'Predicted, PP')
for sim in (simulation, simulation_pp):
    parsed_df = sim[(sim['Ae'] == 0.5) & (sim['Soil']=='sandy-clay')].dropna()
    x = [parsed_df['RateOfChangeDown'].values,0,parsed_df['RateOfChangeUp'].values]
    y = np.repeat(10*(1.5-i),3)
    ax.plot(x,y,'o-',label=labels[i])
    i += 1


# calculate how much of the kde we capture with the simulation
pdf = gaussian_kde(post_cpm['RateOfChange'])#.integrate_box_1d(x[0],x[2])
capture = gaussian_kde(post_cpm['RateOfChange']).integrate_box_1d(x[0],x[2])
custom_x_ticks(ax, my_xticks=np.log10([0.5, 0.75, 1.0, 1.5, 2.0]))
ax.legend(loc='best')
plt.savefig(fig_dir+'rate_of_change_simulations.pdf',dpi=300)
# table 1: simulated rates of change for various Ae and soils


# figure 2: rate of change over the seasons

site_name = ('asu_pp_open','asu_pp_closed','indianapolis')
i=0
for site in (pre_cpm,post_cpm,indianapolis):
    fig, ax = plt.subplots()
    for season in site['Season'].unique():
        parsed_df = site[site['Season']==season]
        sns.kdeplot(parsed_df['RateOfChange'], gridsize=5e3, ax=ax, label=season)
    custom_x_ticks(ax, my_xticks=np.log10([0.5, 0.75, 1.0, 1.5, 2.0]))
    ax.legend(loc='best')
    plt.savefig(fig_dir+'rate_of_change_season_%s.pdf' % site_name[i],dpi=300)
    i+=1

# figure 3: resampled change in concentration
resample_times = ('1D','1W','1M',)
labels = {
    '1h': 'Hourly resampling',
    '1D': 'Daily resampling',
    '1W': 'Weekly resampling',
    '1M': 'Monthly resampling',
}
# subfig: asu
fig, ax = plt.subplots()
for resample_time in resample_times:
    asu, pre_cpm, post_cpm = process_asu(resample_time=resample_time)
    indianapolis = process_indianapolis(resample_time=resample_time)
    sns.kdeplot(asu['RateOfChange'], gridsize=5e3, ax=ax, label=labels[resample_time])

custom_x_ticks(ax, my_xticks=np.log10([0.1, 0.2, 0.5, 1.0, 2, 5, 10.0]))
ax.set_xlabel('$\\frac{\\Delta \\log(c)}{\\Delta \\mathrm{resampling period}}$')
fig.savefig(fig_dir+'resampling_asu.pdf',dpi=300)
# subfig: indianapolis

fig, ax = plt.subplots()
for resample_time in resample_times:
    indianapolis = process_indianapolis(resample_time=resample_time)
    sns.kdeplot(indianapolis['RateOfChange'], gridsize=5e3, ax=ax, label=labels[resample_time])

custom_x_ticks(ax, my_xticks=np.log10([0.5, 0.75, 1.0, 1.5, 2]))
ax.set_xlabel('$\\frac{\\Delta \\log(c)}{\\Delta \\mathrm{resampling period}}$')
fig.savefig(fig_dir+'resampling_indianapolis.pdf',dpi=300)

plt.show()
