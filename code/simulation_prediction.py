import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

df['AirExchangeRate'] *= 3600 # convert from 1/s to 1/hr
df['logIndoorConcentration'] = df['IndoorConcentration'].apply(np.log10)
df['logAttenuationSubslab'] = df['AttenuationSubslab'].apply(np.log10)

p_in = df['IndoorOutdoorPressure']
Ae = df['AirExchangeRate']
alpha = df['logAttenuationSubslab']

# 2d interpolation function
interp_func = interp2d(
    p_in,
    Ae,
    alpha,
    kind='linear',
)

fig, ax = plt.subplots()
asu['logAttenuationSubslab'].plot(kind='kde',ax=ax)


samp_ae = asu['AirExchangeRate'].sample(10).values
samp_p = asu['IndoorOutdoorPressure'].sample(10).values


sim_alpha = interp_func(samp_p, samp_ae)


prediction = pd.DataFrame(
    {
        #'Sampled Ae': samp_ae,
        #'Sampled p': samp_p,
        'Predicted alpha': sim_alpha.flatten(),
    },
)


asu['logAttenuationSubslab'].plot(kind='kde',ax=ax, label='Data')

prediction.plot(kind='kde',ax=ax, label='Prediction')


#ax.plot(samp.values, np.repeat(0.1, len(samp)), 'o')
#samp['AirExchangeRate'].plot(ax=ax)

plt.legend()
plt.show()
