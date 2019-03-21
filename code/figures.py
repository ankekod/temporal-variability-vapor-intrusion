import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy import stats
from get_simulation_data import Soil, PreferentialPathway
import datetime

"""
This figure shows a case (North Island NAS) where indoor/outdoor pressure
difference the key driver at a VI site, and how are are able to model this
by assuming a permeable soil type (sand).
"""
class Figure1:
    def __init__(self,y_data_log=False , norm_conc=True):
        nas = pd.read_csv('./data/north_island.csv').dropna()
        asu = pd.read_csv('./data/asu_house.csv').dropna()
        asu_open = asu.loc[asu['Phase']=='Open']
        asu_closed = asu.loc[asu['Phase']=='Closed']


        x='IndoorOutdoorPressure'

        if y_data_log is True:
            y='logIndoorConcentration'
        else:
            y='IndoorConcentration'


        datasets = (nas, asu_open, asu_closed,)
        labels = ('North Island NAS', 'ASU house, PP open', 'ASU house, PP closed',)

        fig, ax = plt.subplots()

        for data, label in zip(datasets, labels):
            if norm_conc is True:
                if y_data_log is True:
                    data2 = data[y]-data[y].mean()
                else:
                    data2 = data[y]/data[y].mean()

            else:
                data2 = data[y]


            r, p = stats.pearsonr(data[x], data2)

            print(label)
            print(data[x].describe(percentiles=[0.05,0.95]), data['IndoorConcentration'].describe(percentiles=[0.05,0.95]), data2.describe(percentiles=[0.05,0.95]))
            sns.kdeplot(
                data=data[x],
                data2=data2,
                shade_lowest=False,
                shade=True,
                label=label + ', r = %1.2f' % r, # TODO: Add Pearson's r analysis
                ax=ax,
            )



        yticks, yticklabels = get_log_ticks(-1, 1.5)
        print(yticklabels)
        # formatting options
        ax.set(
            xlim=[-30,15],
            ylim=[-1,1.5],
            xlabel='$p_\\mathrm{in/out} \\; \\mathrm{(Pa)}$',
            ylabel='$c_\\mathrm{in}/c_\\mathrm{in,mean}$',
            title='Relationship between indoor/outdoor pressure difference and\nTCE in indoor air (normalized to dataset mean concentration)',
            #yscale='log',
            yticks=yticks,
            yticklabels=yticklabels,
        )
        plt.legend(loc='upper left')
        #plt.savefig('./figures/2d_kde/nas_asu_pp.pdf', dpi=300)
        #plt.savefig('./figures/2d_kde/nas_asu_pp.png', dpi=300)

        plt.show()

        return


def get_log_ticks(start, stop, style='f'): # TODO: Remove the unnecessary ticklabels

    ticks = np.array([])
    for int_now in np.arange(np.floor(start),np.ceil(stop)+1):
        ticks = np.append(ticks, np.arange(0.1,1.1,0.1)*10.0**int_now)


    if style == 'f':
        labels = ['%1.1f' % tick for tick in ticks]
    elif style=='e':
        labels = ['%1.1e' % tick for tick in ticks]
    ticks = np.log10(ticks)


    return ticks, labels


class AttenuationSubslab:
    def __init__(self):
        asu = pd.read_csv('./data/asu_house.csv')
        asu = asu.loc[asu['Phase']!='CPM']

        ax = sns.boxplot(x="Phase", y="logAttenuationSubslab", data=asu)

        ticks, labels = get_log_ticks(-4,2, style='e')
        ax.set(
            xlabel='Preferential pathway status',
            ylabel='Attenuation from subslab',
            #yticks=ticks,
            #yticklabels=labels,
        )

        plt.tight_layout()
        plt.savefig('./figures/temporal_variability/asu_attenuation_subslab.pdf', dpi=300)
        plt.savefig('./figures/temporal_variability/asu_attenuation_subslab.png', dpi=300)

        plt.show()
        return


class Modeling:
    # TODO: Add log ticks
    # TODO: Fix plot colors
    # TODO: Remove fill for PP uniform case
    # TODO: Fix legend labels
    def __init__(self):
        sim = PreferentialPathway().data
        asu = pd.read_csv('./data/asu_house.csv')
        #print(asu['AirExchangeRate'].describe(percentiles=[0.1, 0.9]))
        asu = asu.loc[ (asu['Phase']!='CPM') ]


        # isolates the Ae = 0.5 case for the uniform soil modeling
        sim_data_to_remove = [
            (sim['Simulation']=='Pp Uniform') &
            (
                (sim['AirExchangeRate'] < 0.45) |
                (sim['AirExchangeRate'] > 0.55)
            )
        ]

        sim_data_to_remove = np.invert(sim_data_to_remove)
        sim = sim[sim_data_to_remove[0]]


        # options
        min_Ae, max_Ae = 0.3, 0.9 # the min and max air exchange used for the "fill"

        fig, (ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=True, figsize=[6.4, 6.4], dpi=300)

        pp_max = sim.loc[(sim['Simulation']=='Pp')&(sim['AirExchangeRate']==min_Ae)]
        pp = sim.loc[(sim['Simulation']=='Pp')&(sim['AirExchangeRate']==0.5)]
        pp_min = sim.loc[(sim['Simulation']=='Pp')&(sim['AirExchangeRate']==max_Ae)]


        ax1.plot(pp['IndoorOutdoorPressure'], pp['logAttenuationGroundwater'],label='PP')
        ax1.fill_between(pp['IndoorOutdoorPressure'], pp_min['logAttenuationGroundwater'], pp_max['logAttenuationGroundwater'],alpha=0.5)

        # ax1
        sns.regplot(
            data=asu.loc[asu['Phase']=='Open'],
            x='IndoorOutdoorPressure',
            y='logAttenuationAvgGroundwater',
            ax=ax1,
            fit_reg=False,
            x_bins=np.linspace(-5,5,40),
            ci='sd',
            label='ASU house',
            color=sns.color_palette()[0]
        )
        sns.lineplot(
            data=sim.loc[sim['Simulation']=='Pp Uniform'],
            x='IndoorOutdoorPressure',
            y='logAttenuationGroundwater',
            ax=ax1,
            label='PP, \"uniform\" soil',
        )
        sns.lineplot(
            data=sim.loc[sim['Simulation']=='Pp Uncontaminated'],
            x='IndoorOutdoorPressure',
            y='logAttenuationGroundwater',
            ax=ax1,
            label='Uncontaminated PP'
        )


        no_pp_max = sim.loc[(sim['Simulation']=='No Pp')&(sim['AirExchangeRate']==min_Ae)]
        no_pp = sim.loc[(sim['Simulation']=='No Pp')&(sim['AirExchangeRate']==0.5)]
        no_pp_min = sim.loc[(sim['Simulation']=='No Pp')&(sim['AirExchangeRate']==max_Ae)]

        ax2.plot(no_pp['IndoorOutdoorPressure'], no_pp['logAttenuationGroundwater'], label='No PP')
        ax2.fill_between(no_pp['IndoorOutdoorPressure'], no_pp_min['logAttenuationGroundwater'], no_pp_max['logAttenuationGroundwater'],alpha=0.5)

        # ax2
        sns.regplot(
            data=asu.loc[asu['Phase']=='Closed'],
            x='IndoorOutdoorPressure',
            y='logAttenuationAvgGroundwater',
            ax=ax2,
            fit_reg=False,
            x_bins=np.linspace(-5,5,40),
            ci='sd',
            label='ASU house',
            color=sns.color_palette()[0],
        )

        ax1.set(
            xlabel='',
            ylabel='$\\alpha_\\mathrm{gw}$',
            title='Preferential pathway open',
            xlim=[-5,5],
        )


        ax2.set(
            xlabel='$p_\\mathrm{in/out} \; \\mathrm{(Pa)}$',
            ylabel='$\\alpha_\\mathrm{gw}$',
            title='Preferential pathway closed',
            xlim=[-5,5],
            yticklabels=['%1.1e' % 10.0**tick for tick in ax1.get_yticks()],
        )


        plt.tight_layout()
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        plt.savefig('./figures/simulation_predictions/land_drain_scenarios_combo.pdf')
        plt.savefig('./figures/simulation_predictions/land_drain_scenarios_combo.png')
        plt.show()


        return

class IndianapolisTime:
    def __init__(self):
        data = pd.read_csv('./data/indianapolis.csv')
        data['Time'] = data['Time'].apply(pd.to_datetime)
        data.sort_values(by='Time')

        fig, ax = plt.subplots(dpi=300)

        sns.lineplot(
            data=data[data['Specie']=='Trichloroethene'],
            x='Time',
            y='IndoorConcentration',
            ax=ax,
        )

        ax.set_yscale('log')
        ax.set(
            title='Indoor TCE concentration at the Indianapolis site',
            ylabel='$c_\\mathrm{in} \; \\mathrm{(\\mu g/m^3)}$',
            xlim=([datetime.date(2011, 8, 11), datetime.date(2011, 10, 15)]),
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('./figures/temporal_variability/time_indianapolis.png')
        plt.savefig('./figures/temporal_variability/time_indianapolis.pdf')

        plt.show()
        return

class AirExchangeRateKDE:
    def __init__(self):

        data = pd.read_csv('./data/asu_house.csv').dropna()
        data = data.loc[data['Phase']!='CPM']


        seasons = []
        p_10 = []
        p_50 = []
        p_90 = []
        Ae_10 = []
        Ae_50 = []
        Ae_90 = []

        for season in data['Season'].unique():
            analysis = data.loc[data['Season']==season][['IndoorOutdoorPressure','AirExchangeRate']].describe(percentiles=[0.1,0.9])

            seasons.append(season)
            p_10.append(analysis['IndoorOutdoorPressure']['10%'])
            p_50.append(analysis['IndoorOutdoorPressure']['50%'])
            p_90.append(analysis['IndoorOutdoorPressure']['90%'])
            Ae_10.append(analysis['AirExchangeRate']['10%'])
            Ae_50.append(analysis['AirExchangeRate']['50%'])
            Ae_90.append(analysis['AirExchangeRate']['90%'])

        percentiles = pd.DataFrame({
            'Season': seasons,
            'IndoorOutdoorPressure 10%': p_10,
            'IndoorOutdoorPressure 50%': p_50,
            'IndoorOutdoorPressure 90%': p_90,
            'AirExchangeRate 10%': Ae_10,
            'AirExchangeRate 50%': Ae_50,
            'AirExchangeRate 90%': Ae_90,
        })

        #percentiles.to_csv('./air_exchange_rate.csv', index=False)
        fig, ax = plt.subplots(dpi=300)
        sns.kdeplot(
            data=data['IndoorOutdoorPressure'],
            data2=data['AirExchangeRate'],
            shade_lowest=False,
            shade=True,
        )

        ax.set(
            title='2D KDE showing distributions and relationship between\nindoor/outdoor pressure and air exchange rate\nafter the land drain was closed',
            ylabel='$A_e \; \\mathrm{(1/hr)}$',
            xlabel='$p_\\mathrm{in/out} \; \\mathrm{(Pa)}$',
            ylim=[0,1.75],
            xlim=[-5,5],
        )

        plt.tight_layout()
        plt.savefig('./figures/2d_kde/pressure_air_exchange_rate.png')
        plt.savefig('./figures/2d_kde/pressure_air_exchange_rate.pdf')

        #plt.show()
        return

#Figure1(y_data_log=True,norm_conc=True)
#AttenuationSubslab()
Modeling()
#IndianapolisTime()
#AirExchangeRateKDE()
