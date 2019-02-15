import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.patches as mpatches


db_asu = sqlite3.connect('/Users/duckerfeller/Dropbox/Research/asu_house.db')
db_ind = sqlite3.connect('/Users/duckerfeller/Dropbox/Research/indianapolis.db')

asu = pd.read_sql_query("SELECT day, pressure_difference AS Pressure, tce_emission_rate, building_flow_rate FROM parameters;", db_asu).interpolate(limit_area='inside',limit=250).dropna()
asu['Concentration'] = asu['tce_emission_rate']/asu['building_flow_rate']
cpm_start, cpm_end = 780.0, 1157.0
asu_pre_cpm = asu[(asu['day'] < cpm_start) & (asu['Pressure'] < 7.5)].copy().dropna()
asu_pre_cpm['Pressure'] = -1.0*asu_pre_cpm['Pressure'].dropna()


ind_chloro = pd.read_sql_query(
    "SELECT \
        O.StopDate AS StopDate, \
        AVG(C.Value) AS Concentration, \
        AVG(P.Value) AS Pressure \
    FROM \
        Observation_Status_Data AS O, \
        VOC_Data_SRI_8610_Onsite_GC AS C, \
        Differential_Pressure_Data AS P \
    WHERE \
        O.Value = 'not yet installed' AND \
        C.StopDate = O.StopDate AND \
        C.Location = '422BaseS' AND \
        P.StopDate = O.StopDate AND \
        C.Variable = 'Chloroform' AND \
        P.Variable = 'Basement.Vs.Exterior' AND \
        P.Location = '422' \
    GROUP BY \
        O.StopDate \
    ;", db_ind)

ind_pce = pd.read_sql_query(
    "SELECT \
        O.StopDate AS StopDate, \
        AVG(C.Value) AS Concentration, \
        AVG(P.Value) AS Pressure \
    FROM \
        Observation_Status_Data AS O, \
        VOC_Data_SRI_8610_Onsite_GC AS C, \
        Differential_Pressure_Data AS P \
    WHERE \
        O.Value = 'not yet installed' AND \
        C.StopDate = O.StopDate AND \
        C.Location = '422BaseS' AND \
        P.StopDate = O.StopDate AND \
        C.Variable = 'Tetrachloroethene' AND \
        P.Variable = 'Basement.Vs.Exterior' AND \
        P.Location = '422' \
    GROUP BY \
        O.StopDate \
    ;", db_ind)


asu_pre_cpm.name = "ASU House, TCE"
ind_chloro.name = "Indianapolis, Chloroform"
ind_pce.name = "Indianapolis, PCE"

dfs = []
alpha = 0.6
cmaps = ["Blues", "Reds", "Greens"]

df = pd.DataFrame({'x': [], 'y': []})
g = sns.JointGrid(x="x", y="y", data=df) # empty joint grid
label_patches = []
for i, df in enumerate([asu_pre_cpm, ind_chloro, ind_pce]):
    print("Dataset: %s" % df.name)

    df['Concentration'] = df['Concentration'].apply(pd.to_numeric, errors='ignore').apply(lambda x: np.log10(x)).replace([np.inf, -np.inf], np.nan).dropna()
    df['Concentration'] -= df['Concentration'].mean()

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

g.ax_joint.set_xlabel('Indoor-outdoor pressure difference (Pa)')
g.ax_joint.set_ylabel('Deviation from mean TCE in indoor air')
my_yticks = np.log10([0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0])
my_ytick_labels = ["%.2f$\\cdot\\mu$" % y_tick for y_tick in 10.0**my_yticks]
g.ax_joint.set_xlim((-7.5,5))
g.ax_joint.set_ylim((my_yticks[0],my_yticks[-1]))
g.ax_joint.set_yticks(my_yticks)
g.ax_joint.set_yticklabels(my_ytick_labels)
g.ax_joint.legend(handles=label_patches, loc='upper left')
plt.tight_layout()
dir = '/Users/duckerfeller/Dropbox/Research/Figures/KDE/'
name = 'pp.pdf'

plt.savefig(dir+name, dpi=300,)

#plt.show()
plt.clf()
