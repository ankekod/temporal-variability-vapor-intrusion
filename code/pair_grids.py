import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy import stats



asu = pd.read_csv('./data/asu_house.csv')
indianapolis = pd.read_csv('./data/indianapolis.csv')

# data processing
asu = asu.loc[(asu['IndoorOutdoorPressure'] >= -5) & (asu['IndoorOutdoorPressure'] <= 5)] # TODO: kind of ugly "hack", should instead adjust axes

# global settings
num_bins = 10
fig_path = './figures/pair_grids/'
ext = '.png'
dpi = 300

# method to set custom axis labels
def custom_axis(g):

    replacements = {
        'IndoorOutdoorPressure': '$\\Delta p_\\mathrm{in/out}$ (Pa)',
        'AirExchangeRate': '$A_e$ (1/h)',
        'logIndoorConcentration': '$\\log{(c_\\mathrm{in})}$ ($\\mathrm{\\mu g/m^3}$)',
        'logAttenuationSubslab': '$\\log{(\\alpha_\\mathrm{subslab})}$'
    }

    for i in range(len(g.axes)):
        for j in range(len(g.axes)):
            xlabel = g.axes[i][j].get_xlabel()
            ylabel = g.axes[i][j].get_ylabel()
            if xlabel in replacements.keys():
                g.axes[i][j].set_xlabel(replacements[xlabel])
            if ylabel in replacements.keys():
                g.axes[i][j].set_ylabel(replacements[ylabel])
    return g

"""
# demonstrate the variability

# asu
# PP hue
g = sns.PairGrid(
    asu.loc[asu['Phase']!='CPM'][['IndoorOutdoorPressure','AirExchangeRate','logIndoorConcentration','logAttenuationSubslab','Phase']],
    hue='Phase',
)
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.regplot, x_bins=num_bins, truncate=True, )
g = g.map_diag(sns.kdeplot, shade=True)
g = g.add_legend()
g = custom_axis(g)
plt.savefig(fig_path+'asu_phase'+ext,dpi=dpi)

# season hue
for phase in asu['Phase'].unique():
    try:
        g = sns.PairGrid(
            asu.loc[asu['Phase']==phase][['IndoorOutdoorPressure','AirExchangeRate','logIndoorConcentration','logAttenuationSubslab','Season']],
            hue='Season',
            hue_order=['Winter', 'Fall', 'Spring', 'Summer'],
        )
        g = g.map_upper(plt.scatter)
        g = g.map_lower(sns.regplot, x_bins=num_bins, truncate=True, )
        g = g.map_diag(sns.kdeplot, shade=True)
        g = g.add_legend()
        g = custom_axis(g)
        g.fig.suptitle('Land drain ' + phase.lower())
        plt.savefig(fig_path+'asu_season_'+phase.lower()+ext,dpi=dpi)
    except:
        continue

# indianapolis
# specie hue
g = sns.PairGrid(
    indianapolis[['IndoorOutdoorPressure','logIndoorConcentration','logAttenuationSubslab','Specie']],
    hue='Specie',
)
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.regplot, x_bins=num_bins, truncate=True, )
g = g.map_diag(sns.kdeplot, shade=True)
g = g.add_legend()
g = custom_axis(g)
plt.savefig(fig_path+'indianapolis_species'+ext,dpi=dpi)

# season hue
for specie in indianapolis['Specie'].unique():
    g = sns.PairGrid(
        indianapolis.loc[indianapolis['Specie']==specie][['IndoorOutdoorPressure','logIndoorConcentration','logAttenuationSubslab','Season']],
        hue='Season',
        hue_order=['Winter', 'Fall', 'Summer'],
    )
    g = g.map_upper(plt.scatter)
    g = g.map_lower(sns.regplot, x_bins=num_bins, truncate=True, )
    g = g.map_diag(sns.kdeplot, shade=True)
    g = g.add_legend()
    g = custom_axis(g)
    g.fig.suptitle(specie)
    plt.savefig(fig_path+'indianapolis_season_'+specie.lower()+ext,dpi=dpi)

"""

class SimpleGrid:
    def __init__(self):
        # loads data
        asu = pd.read_csv('./data/asu_house.csv')
        indianapolis = pd.read_csv('./data/indianapolis.csv')
        nas = pd.read_csv('./data/north_island.csv').dropna()

        # processes data
        asu = asu.loc[(asu['Phase']!='CPM')] # & ((asu['IndoorOutdoorPressure'] >= -5) & (asu['IndoorOutdoorPressure'] <= 5))
        indianapolis = indianapolis.loc[indianapolis['Specie']=='Tetrachloroethene']
        # simpler figure for publication
        gs = gridspec.GridSpec(3, 3)



        plt.figure()
        ax1 = plt.subplot(gs[0, 0]) # p distributions ASU
        ax2 = plt.subplot(gs[0, 1]) # - " - Indianapolis
        ax3 = plt.subplot(gs[0, 2]) # - " - nas
        ax4 = plt.subplot(gs[1, 0])
        ax5 = plt.subplot(gs[1, 1])
        ax6 = plt.subplot(gs[1, 2])
        ax7 = plt.subplot(gs[2, 0])

        for phase in asu['Phase'].unique():
            sns.kdeplot(
                data=asu.loc[asu['Phase']==phase]['IndoorOutdoorPressure'],
                shade=True,
                ax=ax1,
                label=phase,
            )
            sns.regplot(
                data=asu.loc[asu['Phase']==phase][['IndoorOutdoorPressure','logIndoorConcentration']],
                x='IndoorOutdoorPressure',
                y='logIndoorConcentration',
                x_bins=np.linspace(-5,5,10),
                ax=ax4,
            )
            sns.regplot(
                data=asu.loc[asu['Phase']==phase][['IndoorOutdoorPressure','AirExchangeRate']],
                x='IndoorOutdoorPressure',
                y='AirExchangeRate',
                x_bins=np.linspace(-5,5,10),
                ax=ax7,
            )

        # indianapolis plots
        sns.kdeplot(
            data=indianapolis['IndoorOutdoorPressure'],
            shade=True,
            ax=ax2,
            legend=False,
        )


        sns.regplot(
            data=indianapolis[['IndoorOutdoorPressure','logIndoorConcentration']],
            x='IndoorOutdoorPressure',
            y='logIndoorConcentration',
            x_bins=np.linspace(-10,10,10),
            ax=ax5,
        )


        # nas plots
        sns.kdeplot(
            data=nas['IndoorOutdoorPressure'],
            shade=True,
            ax=ax3,
            legend=False,
        )
        sns.regplot(
            data=nas[['IndoorOutdoorPressure','logIndoorConcentration']],
            x='IndoorOutdoorPressure',
            y='logIndoorConcentration',
            x_bins=np.arange(nas['IndoorOutdoorPressure'].min(),nas['IndoorOutdoorPressure'].max(),10),
            ax=ax6,
        )


        ax1.set(
            xlim=[-5,5],
            ylabel='Density',
            title='ASU house',
        )

        ax2.set(
            title='Indianapolis duplex',
        )
        ax3.set(
            title='North Island NAS',
        )

        # fix axis labels
        for ax in (ax5,ax6):
            ax.set(xlabel='$\\Delta{p_\\mathrm{in/out}}$ (Pa)',ylabel='')


        ax4.set(
            xlabel='',
            ylabel='$\\log{(c_\\mathrm{in})} \\; \\mathrm{(\\mu g/m^3)}$',
        )
        ax7.set(
            xlabel='$\\Delta{p_\\mathrm{in/out}}$ (Pa)',
            ylabel='$A_e \\; \\mathrm{(1/hr)}$',
            ylim=[0,1],
        )

        # fix axis limits

        # TODO: fix the plot colors in the ASU column
        # TODO: increase resolution of ASU KDE
        ax1.legend()

        plt.tight_layout()
        return


simple_grid = SimpleGrid()
plt.show()
