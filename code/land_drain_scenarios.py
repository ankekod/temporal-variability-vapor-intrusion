import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp2d


# loads data
asu = pd.read_csv('./data/asu_house.csv')
#simulation = pd.read_csv('./data/asu_house.csv')

dfs = []
for file in ('pp','pp_uniform','no_pp',):
    df = pd.read_csv('./data/simulation/sweep_'+file+'.csv',header=4)
    df['Simulation'] = np.repeat(file.replace('_',' ').title(),len(df))
    dfs.append(df)

df = pd.concat(dfs)
df.rename(
    columns={
        '% Ae': 'AirExchangeRate',
        'p_in': 'IndoorOutdoorPressure',
        'Attenuation to groundwater, Global Evaluation: Attenuation to groundwater {gev2}': 'AttenuationGroundwater',
        'Global Evaluation: Relative air entry rate {gev3}': 'RelativeAirEntryRate',
        'Average crack Peclet number, Global Evaluation: Crack Peclet number {gev4}': 'Peclet',
        'TCE in indoor air, Global Evaluation: TCE in indoor air {gev5}': 'IndoorConcentration',
        'Global Evaluation: TCE emission rate {gev6}': 'EntryRate',
        'Attenuation to subbase, Global Evaluation: Attenuation to subslab {gev9}': 'AttenuationSubslab',
        'Global Evaluation: Average TCE in subslab {gev10}': 'SubslabConcentration',
    },
    inplace=True,
)


# data choosing
df['AirExchangeRate'] *= 3600 # convert from 1/s to 1/hr
df['logIndoorConcentration'] = df['IndoorConcentration'].apply(np.log10)
df['logAttenuationSubslab'] = df['AttenuationSubslab'].apply(np.log10)
df['logAttenuationGroundwater'] = df['AttenuationGroundwater'].apply(np.log10)

#fig, ax = plt.subplots()


class ConstantAe:
    def __init__(self):

        g = sns.lmplot(
            data=asu.loc[asu['Phase']!='CPM'][['IndoorOutdoorPressure','logAttenuationAvgGroundwater','Phase']],
            x='IndoorOutdoorPressure',
            y='logAttenuationAvgGroundwater',
            hue='Phase',
            x_bins=np.linspace(-5,5,20),
            fit_reg=False,
            legend_out=False,
            legend=False,
            #aspect=1.5,
        )

        ax = g.axes[0][0]
        sns.lineplot(
            data=df.loc[(df['AirExchangeRate'] >= 0.4) & (df['AirExchangeRate'] <= 0.6)][['IndoorOutdoorPressure','logAttenuationGroundwater','Simulation']],
            x='IndoorOutdoorPressure',
            y='logAttenuationGroundwater',
            hue='Simulation',
            hue_order=['Pp', 'No Pp', 'Pp Uniform',],
            ax=ax,
            #legend=False,
        )


        handles, labels = ax.get_legend_handles_labels()
        handles = handles[1:]

        labels = (
            'PP present',
            'PP absent',
            'PP present, \ngravel sub-base\nabsent',
            'Data, PP open',
            'Data, PP closed',
        )

        ax.legend(
            handles,
            labels,
            #ncol=2,
            loc='best',
        )

        ax.set_xlim([-6,6])
        ax.set_ylim([-7,-3.7])

        ax.set_xlabel('$\\Delta p_\\mathrm{in/out}$ (Pa)')
        ax.set_ylabel('$\\log{(\\alpha_\\mathrm{gw})}$')
        my_ytick_labels = ["%1.0e" % y_tick for y_tick in 10.0**ax.get_yticks()]
        ax.set_yticklabels(my_ytick_labels)
        ax.set_title('Modeling PP scenarios, assuming constant $A_e$,\nand comparing to \"ASU house\" field data')


        plt.tight_layout()

        plt.savefig('./figures/simulation_predictions/land_drain_scenarios_constant_ae.png',dpi=300)
        plt.show()
        return


class FluctuatingAe:
    def __init__(self):
        ae_describe = asu['AirExchangeRate'].describe(percentiles=[0.05, 0.95])
        data = asu.loc[
            ((asu['IndoorOutdoorPressure']>=-5) & (asu['IndoorOutdoorPressure']<=5)) &
            (asu['Phase']!='CPM')
            #(
            #    (asu['logAttenuationAvgGroundwater']>=quantiles[0]) &
            #    (asu['logAttenuationAvgGroundwater']<=quantiles[1])
            #)
        ]


        sim = df.loc[
            (
                (df['AirExchangeRate'] >= ae_describe['5%']) &
                (df['AirExchangeRate'] <= ae_describe['95%'])
            )
        ]
        g = sns.lmplot(
            data=data,
            x='IndoorOutdoorPressure',
            y='logAttenuationAvgGroundwater',
            hue='Phase',
            x_bins=np.linspace(-5,5,20),
            fit_reg=False,
            legend_out=False,
            legend=False,
            #aspect=1.5,
        )

        ax = g.axes[0][0]
        sns.lineplot(
            data=sim,
            x='IndoorOutdoorPressure',
            y='logAttenuationGroundwater',
            hue='Simulation',
            hue_order=['Pp', 'No Pp',],
            ax=ax,
            #legend=False,
        )


        handles, labels = ax.get_legend_handles_labels()
        handles = handles[1:]

        labels = (
            'PP present',
            'PP absent',
            'Data, PP open',
            'Data, PP closed',
        )

        ax.legend(
            handles,
            labels,
            #ncol=2,
            loc='best',
        )

        ax.set_xlim([-6,6])
        ax.set_ylim([-7,-3.7])

        ax.set_xlabel('$\\Delta p_\\mathrm{in/out}$ (Pa)')
        ax.set_ylabel('$\\log{(\\alpha_\\mathrm{gw})}$')
        my_ytick_labels = ["%1.0e" % y_tick for y_tick in 10.0**ax.get_yticks()]
        ax.set_yticklabels(my_ytick_labels)
        ax.set_title('Modeling PP scenarios, assuming constant $A_e$,\nand comparing to \"ASU house\" field data')


        plt.tight_layout()

        plt.savefig('./figures/simulation_predictions/land_drain_scenarios_flucating_ae.png',dpi=300)
        plt.show()
        return
ConstantAe()
FluctuatingAe()
