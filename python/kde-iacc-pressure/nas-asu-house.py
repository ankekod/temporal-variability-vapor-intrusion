import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
from scipy import stats
import matplotlib.patches as mpatches
db_dir = '/home/jonathan/lib/vapor-intrusion-dbs/'


# databases
db_asu = sqlite3.connect(db_dir+'hill-afb.db')
db_nas = sqlite3.connect(db_dir+'north_island_nas.db')

# loads asu data, interpolates and drops NAs
asu = pd.read_sql_query("SELECT * from daily_averages;", db_asu)
phases = pd.read_sql_query("SELECT * from phases;", db_asu)

asu.t = asu.t.apply(pd.to_datetime)
phases.start = phases.start.apply(pd.to_datetime)
phases.stop = phases.stop.apply(pd.to_datetime)

phase1 = phases[(phases.cpm == 'off') & (phases.land_drain == 'open')]
#phase2 = phases[(phases.cpm == 'off') & (phases.land_drain == 'closed')]
phase3 = phases[(phases.cpm == 'off') & (phases.land_drain == 'closed')]


# loads nas data, interpolates and drops NAs
nas = pd.read_sql_query("SELECT tce_indoor_air AS c, pressure_difference AS p FROM north_island_nas;", db_nas).interpolate(limit_area='inside',limit=250).dropna()

cpm_start, cpm_end = 780.0, 1157.0
asu_pre_cpm = asu[asu['t'] < phase1['stop'].values[0]].copy()
asu_post_cpm = asu[asu['t'] > phase3['start'].values[0]].copy()

asu_pre_cpm.name = "ASU House, PP open"
asu_post_cpm.name = "ASU House, PP closed"
nas.name = "North Island"

asu_pre_cpm['c'] = asu_pre_cpm['c'].replace(0.0, np.nan).dropna()

dfs = []
alpha = 0.6
cmaps = ["Blues", "Reds", "Greens"]


label_patches = []
for dp in [False]:
    df = pd.DataFrame({'x': [], 'y': []})
    g = sns.JointGrid(x="x", y="y", data=df) # empty joint grid
    for i, df in enumerate([nas, asu_pre_cpm, asu_post_cpm]):
        dataset = df.name
        print("Dataset: %s" % dataset)

        df['c'] = df['c'].apply(lambda x: np.log10(x)).replace([np.inf, -np.inf], np.nan)
        df['c'] -= df['c'].mean()
        df = df.dropna()
        df['p'] = -1.0*df['p']
        #df[['c','p']].to_csv('~/wtf.csv')

        if dp is True:
            df['p'] -= df['p'].mean()
            name = 'both-deviating'
            xlabel = 'Deviation from mean indoor-outdoor pressure difference (Pa)'
        else:
            name = 'iacc-deviating'
            xlabel = 'Indoor-outdoor pressure difference (Pa)'
        #df = df.dropna()
        r, p = stats.pearsonr(
            df['p'],
            df['c'],
        )
        # bivariate plot
        sns.kdeplot(
            df['p'],
            df['c'],
            cmap=cmaps[i],
            shade=True,
            shade_lowest=False,
            alpha=alpha,
            ax=g.ax_joint,
            )

        color = sns.color_palette(cmaps[i])[2]
        # pressure univariate plot
        sns.kdeplot(
            df['p'],
            color=color,
            shade=True,
            shade_lowest=False,
            alpha=alpha,
            ax=g.ax_marg_x,
            legend=False,
            )
        # concentration univariate plot
        sns.kdeplot(
            df['c'],
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
            label = '%s, r = %1.2f, p = %1.2f' % (dataset, r, p),
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
