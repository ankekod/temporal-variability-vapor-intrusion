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

db = sqlite3.connect(db_dir + 'indianapolis.db')

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
indianapolis['IndoorConcentration'] = indianapolis['IndoorConcentration'].apply(np.log10)

indianapolis.interpolate(inplace=True)
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
indianapolis.name = 'Indianapolis'
alpha = 0.6
cmaps = {
    'Winter': 'Blues',
    'Spring': 'Greens',
    'Summer': 'Reds',
    'Fall': 'Oranges',
}

dp = False

for dataset in (indianapolis,):
    _ = pd.DataFrame({'x': [], 'y': []})
    g = sns.JointGrid(x="x", y="y", data=_) # empty joint grid
    label_patches = []
    for i, season in enumerate(dataset['Season'].unique()):
        #dataset = dataset.name
        df = dataset[dataset['Season']==season]
        print("Dataset: %s, Season: %s" % (dataset.name,season))

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
    my_yticks = np.log10([0.1, 0.5, 1.0, 5.0, 10.0])
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
