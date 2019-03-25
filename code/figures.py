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
class PressureKDE:
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
        plt.savefig('./figures/2d_kde/nas_asu_pp.pdf', dpi=300)
        plt.savefig('./figures/2d_kde/nas_asu_pp.png', dpi=300)

        plt.show()

        return


def get_log_ticks(start, stop, style='e'): # TODO: Remove the unnecessary ticklabels

    ticks = np.array([])
    ints = np.arange(np.floor(start),np.ceil(stop)+1)
    for int_now in ints:
        ticks = np.append(ticks, np.arange(0.1,1.1,0.1)*10.0**int_now)
        ticks = np.unique(ticks)


    if style=='e':
        labels = ['%1.1e' % tick for tick in ticks]
        ticks_to_keep = ['%1.1e' % 10**int for int in ints]
    elif style=='f':
        labels = ['%1.12f' % tick for tick in ticks]
        ticks_to_keep = ['%1.12f' % 10**int for int in ints]

    ticks = np.log10(ticks)

    labels = list(map(lambda x: x.rstrip('0'), labels))
    ticks_to_keep = list(map(lambda x: x.rstrip('0'), ticks_to_keep))

    for i, label in enumerate(labels):

        if label in ticks_to_keep:
            #print('Not removing label')
            continue
        else:
            #print('Removing label')
            labels[i] = ' '
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

        p_vals = sim['IndoorOutdoorPressure'].unique()
        ae = asu['AirExchangeRate'].describe(percentiles=[0.1,0.9])


        new_sim_vals = pd.DataFrame({
            'IndoorOutdoorPressure': np.append(p_vals, p_vals),
            'AirExchangeRate': np.append(),
        })
        #Ae_vals =
        print(new_sim_vals)
        print(analysis[['10%','90%']].values)

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
            ci=95, # 95% confidence interval
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

        ticks, labels = get_log_ticks(-7,-3.5)

        ax1.set(
            xlabel='',
            ylabel='$\\alpha_\\mathrm{gw}$',
            title='Preferential pathway open',
            xlim=[-5,5],
            yticks=ticks,
            yticklabels=labels,
        )


        ax2.set(
            xlabel='$p_\\mathrm{in/out} \; \\mathrm{(Pa)}$',
            ylabel='$\\alpha_\\mathrm{gw}$',
            title='Preferential pathway closed',
            xlim=[-5,5],
            yticks=ticks,
            yticklabels=labels,
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

        # calculates median indoor/outdoor pressure difference and air exchange rate
        # for each season (for later annotation)
        seasons = []
        ps = []
        Aes = []
        for season in data['Season'].unique():
            analysis = data.loc[data['Season']==season][['IndoorOutdoorPressure','AirExchangeRate']].describe(percentiles=[0.1,0.9])
            seasons.append(season)
            ps.append(analysis['IndoorOutdoorPressure']['50%'])
            Aes.append(analysis['AirExchangeRate']['50%'])


        seasonal_medians = pd.DataFrame({'Season': seasons, 'IndoorOutdoorPressure': ps, 'AirExchangeRate': Aes})





        fig, ax = plt.subplots(dpi=300)
        sns.kdeplot(
            data=data['IndoorOutdoorPressure'],
            data2=data['AirExchangeRate'],
            shade_lowest=False,
            shade=True,
        )


        for season, p, Ae in zip(seasons, ps, Aes):
            ax.annotate(
                season,
                xy=(p,Ae),
                xytext=(p*5,Ae+0.6),
                arrowprops=dict(facecolor='black', shrink=0.001),
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

        plt.show()
        return

class Diurnal:
    def __init__(self):
        from scipy.interpolate import CubicSpline
        path = './data/diurnal/simulation_results/'

        p_diurnal = pd.read_csv('./data/diurnal/pressure.csv')
        ae_diurnal = pd.read_csv('./data/diurnal/air_exchange_rate.csv')
        pp_closed_const_ae = pd.read_csv(path+'pp_closed_const_ae.csv',header=4)
        pp_closed = pd.read_csv(path+'pp_closed.csv',header=4)
        pp_open_const_ae = pd.read_csv(path+'pp_open_const_ae.csv',header=4)
        pp_open = pd.read_csv(path+'pp_open.csv',header=4)

        maxmin = lambda x: x['AttenuationGroundwater (1)'].max()/x['AttenuationGroundwater (1)'].min()
        print(
            maxmin(pp_closed_const_ae),
            maxmin(pp_closed),
            maxmin(pp_open_const_ae),
            maxmin(pp_open),
        )
        fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2, dpi=150)
        # TODO: Create interpolate function for PP closed cases to make nicer plots
        x_smooth = np.linspace(0,23,100)
        y_smooth = CubicSpline(p_diurnal['Time'], p_diurnal['IndoorOutdoorPressure'])(x_smooth)
        ax1.plot(x_smooth, y_smooth)



        x_smooth = np.linspace(0,23,200)
        y_smooth = CubicSpline(ae_diurnal['Time'], ae_diurnal['AirExchangeRate'])(x_smooth)
        ax2.plot(x_smooth, y_smooth, label='Diurnal Ae')


        ax2.plot([0,23],[0.5,0.5], label='Constant Ae')

        pp_open_const_ae.plot(
            x='% Time (h)',
            y='AttenuationGroundwater (1)',
            ax=ax3,
            logy=True,
            label='Max change = %1.2f' % maxmin(pp_open_const_ae),
        )

        pp_open.plot(
            x='% Time (h)',
            y='AttenuationGroundwater (1)',
            ax=ax3,
            logy=True,
            label='Max change = %1.2f' % maxmin(pp_open),
        )


        x_smooth = np.linspace(0,23.6,100)
        y_smooth = CubicSpline(pp_closed['% Time (h)'], pp_closed['AttenuationGroundwater (1)'])(x_smooth)
        ax4.semilogy(x_smooth, y_smooth, label='Max change = %1.2f' % maxmin(pp_closed))

        ax4.semilogy(pp_closed_const_ae['% Time (h)'], pp_closed_const_ae['AttenuationGroundwater (1)'], label='Max change = %1.2f' % maxmin(pp_closed_const_ae))


        ax2.legend(loc='best')
        ax4.legend(loc='best')

        ylims = [5e-6, 1e-4]

        ax1.set(
            ylabel='$p_\\mathrm{in/out} \\; \\mathrm{(Pa)}$',
            title='Simulation input:\nMedian diurnal $p_\\mathrm{in/out}$',
        )

        ax2.set(
            ylabel='$A_e \\; \\mathrm{(1/hour)}$',
            title='Simulation input:\nMedian diurnal $A_e$',
        )

        ax3.set(
            ylim=ylims,
            ylabel='$\\alpha_\\mathrm{gw}$',
            xlabel='Time (hour)',
            title='Simulation result:\nPP open cases'
        )
        ax4.set(
            ylim=ylims,
            xlabel='Time (hour)',
            title='Simulation result:\nPP closed cases'
        )

        plt.tight_layout()

        plt.savefig('./figures/simulation_predictions/diurnal.png')
        plt.savefig('./figures/simulation_predictions/diurnal.pdf')
        plt.show()

        return


#Diurnal()
#PressureKDE(y_data_log=True,norm_conc=True)
#AttenuationSubslab()
Modeling()
#IndianapolisTime()
#AirExchangeRateKDE()

"""
df = pd.DataFrame({'x': [0,2], 'y': [1,4]})
df2 = pd.DataFrame({'x':[1,]})

df.append(df2)
df = df.append(df2)

df.sort_values(by='x',inplace=True)

df.interpolate(inplace=True)
print(df)
"""
