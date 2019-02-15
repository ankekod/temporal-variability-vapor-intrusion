import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.patches as mpatches



class Observation:
    def __init__(self,select,where,grouby):
        return
    def opt_select(self,value):
        return
class IndoorAir:
    def __init__(self,method,compound,location,groupby):
        return

class Indianapolis:
    def __init__(self):
        # loads database
        self.db = sqlite3.connect('/home/jonathan/lib/vapor-intrusion-dbs/indianapolis.db')
        #self.db = sqlite3.connect('D:\Dropbox\Research\indianapolis.db')

        return

    def ssd(self,status='not yet installed'):
        select = "O.StopDate AS StopDate"
        table = "Observation_Status_Data AS O"
        where = "O.Value = \"%s\"" % status
        return select, table, where

    def pressure(self,location='422',type='Basement.Vs.Exterior',avg=True):
        if avg is True:
            select = "AVG(P.Value) AS Pressure"
        else:
            select = "P.Value AS Pressure"
        table = "Differential_Pressure_Data AS P"
        where = "P.Location = \"%s\"" % location + " AND " + "P.Variable = \"%s\"" % type

        return select, table, where

    def concentration(self,location='422BaseS',compound='Chloroform',avg=True):
        if avg is True:
            select = "AVG(C.Value) AS Concentration"
        else:
            select = "C.Value AS Concentration"
        table = "VOC_Data_SRI_8610_Onsite_GC AS C"
        where = "C.Location = \"%s\"" % location + " AND " + "C.Variable = \"%s\"" % compound

        return select, table, where



    def sql_query(self,methods):
        select = "SELECT "
        tables = "FROM "
        wheres = "WHERE "


        for method in methods:
            select += method[0] + ", "
            tables += method[1] + ", "
            wheres += method[2] + " AND "

        select = select.rstrip(", ")
        tables = tables.rstrip(", ")
        wheres += "C.StopDate = O.StopDate AND P.StopDate = O.StopDate"
        #wheres = wheres.rstrip(" AND ")
        query = select + " " + tables + " " + wheres + " GROUP BY O.StopDate;"
        df = pd.read_sql_query(query,self.db)
        #print(query)
        return df


locations = {
    'Pressure': ['422','420'],
    'Concentration': ['SGP2', '422BaseS', 'SSP-2', 'SSP-4', 'SGP11', 'SSP-7', '420First', 'SGP9', 'WP-3', 'SGP8', '422First', '420BaseS', 'Outside'],
}

differentials = ['Wall.Vs.Basement', 'SubSlab.Vs.Basement', 'Basement.Vs.Upsatairs', 'DeepSoilgas.Vs.ShallowSoilgas', 'Basement.Vs.Exterior']
#
# TCE works, PCE funkar, Chloroform funkar
#'Trichloroethene',
compounds = ['Tetrachloroethene', 'Chloroform',]

alpha = 0.6
cmaps = ["Blues", "Reds", "Greens"]

locations = ['422',]

for location in locations:
    df = pd.DataFrame({'x': [], 'y': []})
    g = sns.JointGrid(x="x", y="y", data=df) # empty joint grid
    label_patches = []
    for i, compound in enumerate(compounds):
        ind = Indianapolis()

        df = ind.sql_query(
            [ind.ssd(),ind.pressure(location=location),ind.concentration(location=location+'BaseS',compound=compound)]
        )
        print("Dataset:\nLocation: %s\nCompound=%s" % (location, compound))
        df['Concentration'] = df['Concentration'].apply(pd.to_numeric, errors='ignore').apply(lambda x: np.log10(x)).replace([np.inf, -np.inf], np.nan).dropna()
        df['Concentration'] -= df['Concentration'].mean()
        df['Pressure'] = df['Pressure'].dropna()
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
            label = '%s, r = %1.2f, p = %1.2f' % (compound, r, p),
        )
        label_patches.append(label_patch)

    g.ax_joint.set_xlabel('Indoor-outdoor pressure difference (Pa)')
    g.ax_joint.set_ylabel('Deviation from mean IACC')
    my_yticks = np.log10([0.1, 0.5, 1.0, 5.0, 10.0,])
    my_ytick_labels = ["%.1f$\\cdot\\mu$" % y_tick for y_tick in 10.0**my_yticks]
    g.ax_joint.set_ylim((my_yticks[0],my_yticks[-1]))
    g.ax_joint.set_yticks(my_yticks)
    g.ax_joint.set_yticklabels(my_ytick_labels)
    g.ax_joint.legend(handles=label_patches, loc='upper left')
    plt.tight_layout()
    dir = './figures/kde-iacc-pressure/'
    name = 'indianapolis-%s.pdf' % location

    plt.savefig(dir+name, dpi=300,)

    #plt.show()
    plt.clf()





"""
420:
SGP11, SSP7, 420First, 420BaseS

422:
SSP2, SGP8, SGP9, SSP4, WP-3,

def load_data(self):
    df = pd.read_sql_query(
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
            C.Variable = 'Chloroform' AND \
            C.Location = '422BaseS' AND \
            P.StopDate = O.StopDate AND \
            P.Variable = 'Basement.Vs.Exterior' AND \
            P.Location = '422' \
        GROUP BY \
            O.StopDate \
        ;", self.db)
    return df

"""
