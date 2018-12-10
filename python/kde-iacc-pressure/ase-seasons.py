import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import LSODA
from scipy.interpolate import interp1d
import sqlite3
import seaborn as sns
from scipy import stats
import matplotlib.patches as mpatches


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
pre_cpm.name = "ASU House, PP open"
post_cpm.name = "ASU House, PP closed"

alpha = 0.6
cmaps = {
    'Winter': 'Blues',
    'Spring': 'Greens',
    'Summer': 'Reds',
    'Fall': 'Oranges',
}

dp = False

for dataset in (pre_cpm,post_cpm,):
    _ = pd.DataFrame({'x': [], 'y': []})
    g = sns.JointGrid(x="x", y="y", data=_) # empty joint grid
    label_patches = []
    for i, season in enumerate(dataset['Season'].unique()):
        #dataset = dataset.name
        df = dataset[dataset['Season']==season]
        print("Dataset: %s, Season: %s" % (dataset.name,season))
        # TODO: use dataset mean instead of seasonal mean
        df['IndoorConcentration'] -= df['IndoorConcentration'].mean()

        if dp is True:
            df['p'] -= df['p'].mean()
            name = 'both-deviating'
            xlabel = 'Deviation from mean indoor-outdoor pressure difference (Pa)'
        else:
            name = 'iacc-deviating'
            xlabel = 'Indoor-outdoor pressure difference (Pa)'
        #df = df.dropna()
        r, p = stats.pearsonr(
            df['Pressure'],
            df['IndoorConcentration'],
        )
        # bivariate plot
        sns.kdeplot(
            df['Pressure'],
            df['IndoorConcentration'],
            cmap=cmaps[season],
            shade=True,
            shade_lowest=False,
            alpha=alpha,
            ax=g.ax_joint,
            )

        color = sns.color_palette(cmaps[season])[2]
        # pressure univariate plot
        sns.kdeplot(
            df['Pressure'],
            color=color,
            shade=True,
            shade_lowest=False,
            alpha=alpha,
            ax=g.ax_marg_x,
            legend=False,
            )
        # concentration univariate plot
        sns.kdeplot(
            df['IndoorConcentration'],
            color=color,
            shade=True,
            shade_lowest=False,
            alpha=alpha,
            ax=g.ax_marg_y,
            vertical=True,
            legend=False,
            )

        label_patch = mpatches.Patch(
            color = sns.color_palette(cmaps[season])[2],
            label = '%s, r = %1.2f, p = %1.2f' % (season, r, p),
        )
        label_patches.append(label_patch)

    g.ax_joint.set_xlabel(xlabel)
    g.ax_joint.set_ylabel('Deviation from mean TCE in indoor air')
    my_yticks = np.log10([0.01,0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100])
    my_ytick_labels = ["%.2f$\\cdot\\mu$" % y_tick for y_tick in 10.0**my_yticks]
    g.ax_joint.set_ylim((my_yticks[0],my_yticks[-1]))
    g.ax_joint.set_yticks(my_yticks)
    g.ax_joint.set_yticklabels(my_ytick_labels)
    g.ax_joint.legend(handles=label_patches, loc='upper left')
    plt.tight_layout()
    plt.savefig('./figures/kde-iacc-pressure/%s-seasons.pdf' % dataset.name.replace(',','').replace(' ','_').lower(), dpi=300)
    # TODO: dynamically change axis limits?
    #plt.show()
    plt.clf()
    plt.cla()
