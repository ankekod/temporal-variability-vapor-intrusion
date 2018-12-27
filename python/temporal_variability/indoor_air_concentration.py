import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import LSODA
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import sqlite3
import seaborn as sns

# TODO: add indianapolis data and add to present analysis

data_dir = './data/preferential-pathway-sensitivity/'
db_dir = '/home/jonathan/Dropbox/var/'
#db_dir = 'C:\\Users\\jstroem\\Dropbox\\var\\'

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

    pp_status = lambda x: 'Open' if x < phase1['StopTime'].values[0] else (' Closed' if x > phase3['StartTime'].values[0] else 'CPM')
    asu['PP'] = asu['StopTime'].apply(pp_status)



    return asu, pre_cpm, post_cpm

def process_indianapolis(resample_time=False,specie='Tetrachloroethene'):
    indianapolis = pd.read_sql_query(
        "SELECT C.StopDate, C.StopTime AS TimeStamp, C.Value AS Concentration, C.Variable as Specie FROM VOC_Data_SRI_8610_Onsite_GC C, Observation_Status_Data O WHERE DATE(C.StopDate)=DATE(O.StopDate) AND O.Value='not yet installed' AND (C.Location='422BaseS' OR C.Location='422BaseN');", db_indianapolis, )

    indianapolis = indianapolis.assign(StopTime=lambda x: x['StopDate']+' '+x['TimeStamp'])
    indianapolis.drop(columns=['StopDate','TimeStamp'],inplace=True)
    indianapolis['StopTime'] = indianapolis['StopTime'].apply(pd.to_datetime)
    indianapolis.sort_values(by=['StopTime','Specie'],inplace=True)
    if resample_time != False:
        indianapolis = indianapolis.resample(resample_time, on='StopTime', kind='timestamp').mean().reset_index()

    indianapolis['Concentration'] = indianapolis['Concentration'].apply(np.log10).replace([np.inf,-np.inf],np.nan)
    indianapolis = indianapolis.dropna().reset_index(drop=True)

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
    my_xtick_labels = ["%1.3f" % x_tick for x_tick in 10.0**ax.get_xticks()]
    #ax.set_xlim((my_xticks[0], my_xticks[-1]))
    #ax.set_xticks(my_xticks)
    ax.set_xticklabels(my_xtick_labels)
    return




sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
# color palette

def plot_facetgrid(df):
    g = sns.FacetGrid(df, row="Category", hue="Category", aspect=12, height=.75, )
    g.map(sns.kdeplot, "Concentration", clip_on=False, shade=True)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0.65, .25, label, fontweight="bold", color=sns.desaturate(color, 0.5),
                ha="left", va="center", transform=ax.transAxes)
    g.map(label, "Concentration")
    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.25)
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    g.axes[-1][0].set_xlabel('Contaminant in Indoor Air $\\mathrm{(\\mu g/m^3)}$')
    custom_x_ticks(g.axes[-1][0],my_xticks=np.log10([0.001,0.01,0.1,1,10,100]))
    return g

asu, pre_cpm, post_cpm = process_asu()
asu = asu[asu['PP'] != 'CPM'].reset_index() # removes cpm period
asu['Category'] = 'PP ' + asu['PP'] + ', ' + asu['Season']
sns.set_palette(('Blue','Green','Red','Orange','Orange','Blue','Green','Red',))
g = plot_facetgrid(asu)
g.axes[-1][0].set_xlabel('TCE in Indoor Air $\\mathrm{(\\mu g/m^3)}$')
plt.savefig('./figures/temporal_variability/asu.pdf',dpi=300)

indianapolis = process_indianapolis()
indianapolis['Category'] = indianapolis['Specie'] + ', ' + indianapolis['Season']
sns.set_palette(('Red','Red','Red','Orange','Orange','Orange','Blue','Blue','Blue',))
g = plot_facetgrid(indianapolis)
plt.savefig('./figures/temporal_variability/indianapolis.pdf',dpi=300)

#plt.show()

resampled = pd.DataFrame({'Resampling': [], 'Delta': [], 'Dataset': []})

datasets = (
    pre_cpm,
    post_cpm,
    indianapolis[indianapolis['Specie']=='Trichloroethene'],
    indianapolis[indianapolis['Specie']=='Tetrachloroethene'],
    indianapolis[indianapolis['Specie']=='Chloroform'],
)
names = (
    'ASU House, PP Open',
    'ASU House, PP Closed',
    'Indianapolis, TCE',
    'Indianapolis, PCE',
    'Indianapolis, Chloroform',
)

for dataset, name in zip(datasets,names):
    for resampling_time in ('1D','1W','3W','1M',):
        r = dataset.resample(resampling_time ,on='StopTime', kind='timestamp')
        r = r['Concentration'].agg([np.mean, np.max, np.min, np.std])
        to_be_appended = pd.DataFrame({
        'Resampling': np.repeat(resampling_time,len(r)),
        'Delta': r['amax'].values-r['amin'].values,
        'Dataset': np.repeat(name,len(r)),
        })
        resampled = resampled.append(to_be_appended,ignore_index=True)

resampled.dropna(inplace=True)


sns.set_palette(sns.color_palette("muted"))


g = sns.catplot(x="Resampling", y="Delta", hue='Dataset', kind="point", data=resampled)

my_ytick_labels = ["%1.1f" % y_tick for y_tick in 10.0**g.ax.get_yticks()]
g.ax.set_yticklabels(my_ytick_labels)
g.ax.set_xlabel('Resampling Period')
g.ax.set_ylabel('$c_\\mathrm{max}/c_\\mathrm{min}$')
plt.savefig('./figures/temporal_variability/resampling.pdf',dpi=300)
