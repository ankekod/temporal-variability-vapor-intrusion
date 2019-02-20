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
    alpha = df['logAttenuationSubslab']

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
    ax1.set_title('Indoor/outdoor pressure')
    asu['IndoorOutdoorPressure'].plot(kind='kde',ax=ax1, label='Data')
    ax1.plot(samp_p, np.repeat(0, len(samp_p)),'o',label='Sampled data')
    ax1.set_xlim([-15,15])
    # air exchange rate plot
    ax2 = plt.subplot(gs[0, 1]) # row 0, col 1
    ax2.set_title('Air exchange rate')
    asu['AirExchangeRate'].plot(kind='kde',ax=ax2, label='Data')
    ax2.plot(samp_ae, np.repeat(0, len(samp_ae)),'o',label='Sampled data')
    # prediction plot
    ax3 = plt.subplot(gs[1, :]) # row 1, span all columns
    asu['logAttenuationSubslab'].plot(kind='kde',ax=ax3, label='Data')
    prediction.plot(kind='kde',ax=ax3, label='Prediction')
    ax3.set_xlim(-4,2)

    plt.suptitle(sim_case)
    plt.legend()
    return



# data choosing


df['AirExchangeRate'] *= 3600 # convert from 1/s to 1/hr
df['logIndoorConcentration'] = df['IndoorConcentration'].apply(np.log10)
df['AttenuationSubslab'] *= 2e3 # this seems to fix the problem... any way to get it? TODO: Looking into this more
df['logAttenuationSubslab'] = df['AttenuationSubslab'].apply(np.log10)

sim_cases = ('Pp','No Pp',)
phases = ('Open','Closed',)
print(df['Simulation'].unique())
num_samples = 20
for sim_case, phase in zip(sim_cases, phases):
    df_sort = df.loc[df['Simulation']==sim_case] # sorts simulation types
    asu_sort = asu.loc[asu['Phase']==phase]


    interp_func = get_interp_func(df_sort)
    samp_p, samp_ae, prediction = get_prediction(asu_sort)
    make_plots(asu_sort)

plt.show()
