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

# input data for interpolation
def get_interp_func(df):
    p_in = df['IndoorOutdoorPressure']
    Ae = df['AirExchangeRate']
    alpha = df['logAttenuationGroundwater']

    # TODO: increase pressurization limits in simulation
    # 2d interpolation function
    interp_func = interp2d(
        p_in,
        Ae,
        alpha,
        kind='linear',
    )

    return interp_func

def get_prediction(asu):
    # selects p_in and Ae from data distribution and gets predicted indoor air concentration from simulation
    samp_ae = asu['AirExchangeRate'].sample(num_samples).values
    samp_p = asu['IndoorOutdoorPressure'].sample(num_samples).values
    sim_alpha = interp_func(samp_p, samp_ae)

    # puts predicted IACC in dataframe
    prediction = pd.DataFrame(
        {
            #'Sampled Ae': samp_ae,
            #'Sampled p': samp_p,
            'Predicted alpha': sim_alpha.flatten(),
        },
    )

    return samp_p, samp_ae, prediction

def make_plots(asu):
    # plotting
    # Create 2x2 sub plots
    gs = gridspec.GridSpec(2, 2)

    # indoor/outdoor pressure plot
    plt.figure()
    ax1 = plt.subplot(gs[0, 0]) # row 0, col 0
    ax1.set_xlabel('$\\Delta p_\\mathrm{in/out}$ (Pa)')
    asu['IndoorOutdoorPressure'].plot(kind='kde',ax=ax1, label='Data')
    ax1.plot(samp_p, np.repeat(0, len(samp_p)),'o',label='Sampled data')
    ax1.set_xlim([-8,8])
    # air exchange rate plot
    ax2 = plt.subplot(gs[0, 1]) # row 0, col 1
    ax2.set_xlabel('$A_e$ (1/h)')
    asu['AirExchangeRate'].plot(kind='kde',ax=ax2, label='Data')
    ax2.plot(samp_ae, np.repeat(0, len(samp_ae)),'o',label='Sampled data')
    ax2.legend()
    # prediction plot
    ax3 = plt.subplot(gs[1, :]) # row 1, span all columns
    asu['logAttenuationAvgGroundwater'].plot(kind='kde',ax=ax3, label='Data')
    prediction['Predicted alpha'].plot(kind='kde',ax=ax3, label='Prediction')
    ax3.set_xlim([-7,-3])
    my_xtick_labels = ["%1.0e" % x_tick for x_tick in 10.0**ax3.get_xticks()]
    ax3.set_xticklabels(my_xtick_labels)
    ax3.set_xlabel('$\\log{(\\alpha_\\mathrm{gw})}$')
    ax3.legend()

    titles = {
        'Pp': 'Preferential pathway open',
        'No Pp': 'Preferential pathway closed',
    }

    plt.suptitle(titles[sim_case],y=1.0)
    plt.tight_layout()

    return



# data choosing
df['AirExchangeRate'] *= 3600 # convert from 1/s to 1/hr
df['logIndoorConcentration'] = df['IndoorConcentration'].apply(np.log10)
#df['AttenuationSubslab'] *= 2e3 # this seems to fix the problem... any way to get it? TODO: Looking into this more
df['logAttenuationSubslab'] = df['AttenuationSubslab'].apply(np.log10)
df['logAttenuationGroundwater'] = df['AttenuationGroundwater'].apply(np.log10)

# figure saving settings
fig_dir = './figures/simulation_predictions/'
ext = '.pdf'
dpi = 300

"""
sim_cases = ('Pp','No Pp',)
phases = ('Open','Closed',)
num_samples = 15
for sim_case, phase in zip(sim_cases, phases):
    df_sort = df.loc[df['Simulation']==sim_case] # sorts simulation types
    asu_sort = asu.loc[asu['Phase']==phase]

    interp_func = get_interp_func(df_sort)
    samp_p, samp_ae, prediction = get_prediction(asu_sort)
    make_plots(asu_sort)
    plt.savefig(fig_dir+'sampling_simulation_pp_'+phase.lower()+ext,dpi=dpi)



# seasonal analysis
seasons = asu['Season'].unique()
sim_cases = ('Pp','No Pp',)
phases = ('Open','Closed',)
num_samples = 20
for season in seasons:
    for sim_case, phase in zip(sim_cases, phases):
        try:
            df_sort = df.loc[df['Simulation']==sim_case] # sorts simulation types
            asu_sort = asu.loc[(asu['Phase']==phase) & (asu['Season']==season)]

            interp_func = get_interp_func(df_sort)
            samp_p, samp_ae, prediction = get_prediction(asu_sort)
            make_plots(asu_sort)
            titles = {
                'Pp': 'Preferential pathway open',
                'No Pp': 'Preferential pathway closed',
            }

            plt.suptitle(titles[sim_case]+' - '+season.lower(),y=1.0)
            plt.savefig(fig_dir+'sampling_simulation_pp_'+phase.lower()+'_'+season.lower()+ext,dpi=dpi)
        except:
            continue

"""
# "Ae-span" figure


class SimPrediction:
    def __init__(self,status='Closed'):

        options = {
            'Open': ['Open', 'Pp'],
            'Closed': ['Closed', 'No Pp'],
        }
        asu_span = asu.loc[(asu['Phase']==options[status][0]) & ( (asu['IndoorOutdoorPressure']>=-5) & (asu['IndoorOutdoorPressure']<=5) )]
        sim_span = df[df['Simulation']==options[status][1]]

        grp = asu_span[['IndoorOutdoorPressure','AirExchangeRate']].groupby(pd.cut(asu_span['IndoorOutdoorPressure'], np.arange(-5, 5.5, 0.5))).describe(percentiles=[0.05, 0.95])
        print(grp['IndoorOutdoorPressure'])
        p_in = grp['IndoorOutdoorPressure'].index
        p_in = np.arange(-5, 5, 0.5)
        #print(p_in.categories.astype('float'))

        Ae_min = grp[('AirExchangeRate','5%')].values
        Ae_max = grp[('AirExchangeRate','95%')].values
        Ae_mean = grp[('AirExchangeRate','mean')].values

        alpha_min = []
        alpha_max = []
        alpha_mean = []

        func = self.get_interp_func(sim_span)
        for p, min, max, mean in zip(p_in, Ae_min, Ae_max, Ae_mean):

            alpha_min.append(func(p,max)[0])
            alpha_max.append(func(p,min)[0])
            alpha_mean.append(func(p,mean)[0])

        alpha_min = np.array(alpha_min)
        alpha_max = np.array(alpha_max)
        alpha_mean = np.array(alpha_mean)

        fig, ax = plt.subplots()
        #ax.plot(p_in, alpha_min, label='min')
        #ax.plot(p_in, alpha_max, label='max')
        """
        ax.errorbar(
            p_in,
            alpha_mean,
            yerr=(alpha_mean-alpha_min, alpha_max-alpha_mean),
            #xerr=None,
        )
        """

        ax.plot(p_in, alpha_mean, 'k-')
        ax.fill_between(
            p_in,
            alpha_mean-(alpha_mean-alpha_min),
            alpha_mean+(alpha_max-alpha_mean),
        )

        sns.scatterplot(
            data=asu_span.loc[asu_span['IndoorConcentration']>=0.001],
            x='IndoorOutdoorPressure',
            y='logAttenuationAvgGroundwater',
            ax=ax,
            x_bins=p_in,
            #legend=False,
            label='Data',
        )


        plt.legend()
        plt.show()
        #print(grp[('AirExchangeRate','min')].values)

        return

    def get_interp_func(self, df):

        p_in = df['IndoorOutdoorPressure']
        Ae = df['AirExchangeRate']
        alpha = df['logAttenuationGroundwater']
        func = interp2d(
            p_in,
            Ae,
            alpha,
            kind='linear',
        )

        return func



gs = gridspec.GridSpec(2, 2)

sim_pred = SimPrediction(status='Closed')
