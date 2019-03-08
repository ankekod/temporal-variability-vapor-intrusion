import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy import stats

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
        plt.savefig(fig_path+'simple_grid'+ext,dpi=300)
        plt.tight_layout()
        return


simple_grid = SimpleGrid()
plt.show()
