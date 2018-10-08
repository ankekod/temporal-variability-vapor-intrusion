import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
from scipy import stats
import matplotlib.patches as mpatches
db_dir = '/home/jonathan/lib/vapor-intrusion-dbs/'


# databases
db_asu = sqlite3.connect(db_dir+'asu_house.db')
db_nas = sqlite3.connect(db_dir+'north_island_nas.db')

# loads asu data, interpolates and drops NAs
asu = pd.read_sql_query("SELECT day, pressure_difference AS Pressure, tce_emission_rate, building_flow_rate FROM parameters;", db_asu).interpolate(limit_area='inside',limit=250).dropna()

# loads nas data, interpolates and drops NAs
nas = pd.read_sql_query("SELECT tce_indoor_air AS Concentration, pressure_difference AS Pressure FROM north_island_nas;", db_nas).interpolate(limit_area='inside',limit=250).dropna()


asu['Concentration'] = asu['tce_emission_rate']/asu['building_flow_rate']

cpm_start, cpm_end = 780.0, 1157.0

asu_pre_cpm = asu[(asu['day'] < cpm_start) & (asu['Pressure'] < 7.5)].copy().dropna()
asu_post_cpm = asu[asu['day'] > cpm_end].copy().dropna()

asu_pre_cpm.name = "ASU House, PP open"
asu_post_cpm.name = "ASU House, PP closed"
nas.name = "North Island"

dfs = []
alpha = 0.6
cmaps = ["Blues", "Reds", "Greens"]


label_patches = []

for dp in [False]:
    df = pd.DataFrame({'x': [], 'y': []})
    g = sns.JointGrid(x="x", y="y", data=df) # empty joint grid
    for i, df in enumerate([nas, asu_pre_cpm, asu_post_cpm]):
        print("Dataset: %s" % df.name)

        df['Concentration'] = df['Concentration'].apply(pd.to_numeric, errors='ignore').apply(lambda x: np.log10(x)).replace([np.inf, -np.inf], np.nan).dropna()
        df['Concentration'] -= df['Concentration'].mean()
        df['Pressure'] = -1.0*df['Pressure'].dropna()
        if dp is True:
            df['Pressure'] -= df['Pressure'].mean()
            name = 'both-deviating'
            xlabel = 'Deviation from mean indoor-outdoor pressure difference (Pa)'
        else:
            name = 'iacc-deviating'
            xlabel = 'Indoor-outdoor pressure difference (Pa)'
        #df = df.dropna()
        r, p = stats.pearsonr(
            df['Pressure'],
            df['Concentration'],
        )
        # bivariate plot
        sns.kdeplot(
            df['Pressure'],
            df['Concentration'],
            cmap=cmaps[i],
            shade=True,
            shade_lowest=False,
            alpha=alpha,
            ax=g.ax_joint,
            )

        color = sns.color_palette(cmaps[i])[2]
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
            df['Concentration'],
            color=color,
            shade=True,
            shade_lowest=False,
            alpha=alpha,
            ax=g.ax_marg_y,
            vertical=True,
            legend=False,
            )

        label_patch = mpatches.Patch(
            color = sns.color_palette(cmaps[i])[2],
            label = '%s, r = %1.2f, p = %1.2f' % (df.name, r, p),
        )
        label_patches.append(label_patch)

    g.ax_joint.set_xlabel(xlabel)
    g.ax_joint.set_ylabel('Deviation from mean TCE in indoor air')
    #my_yticks = np.arange(-1.0,3, 1)
    my_yticks = np.log10([0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0])
    my_ytick_labels = ["%.2f$\\cdot\\mu$" % y_tick for y_tick in 10.0**my_yticks]
    #g.ax_joint.set_xlim((-28,18))
    g.ax_joint.set_ylim((my_yticks[0],my_yticks[-1]))
    g.ax_joint.set_yticks(my_yticks)
    g.ax_joint.set_yticklabels(my_ytick_labels)
    g.ax_joint.legend(handles=label_patches, loc='upper left')
    plt.tight_layout()
    fig_dir = './figures/kde-iacc-pressure/'

    plt.savefig(fig_dir+'nas-asu-house-'+name+'.pdf', dpi=300,)

    #plt.show()
    plt.clf()
    plt.cla()
