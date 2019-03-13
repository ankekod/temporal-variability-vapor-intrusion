import numpy as np
import pandas as pd

# TODO: Create class with common functions

class PreferentialPathway:
    def __init__(self):
        dfs = []
        for file in ('pp','pp_uniform','no_pp','pp_uncontaminated',):
            df = pd.read_csv('./data/simulation/sweep_'+file+'.csv',header=4)
            df['Simulation'] = np.repeat(file.replace('_',' ').title(),len(df))
            if file == 'pp_uncontaminated':
                df['% Ae'] = np.repeat(0.5/3600, len(df))
                df.rename(columns={'% p_in': 'p_in'},inplace=True)
            dfs.append(df)

        df = pd.concat(dfs,sort=True)
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


        self.data = df
        return

class Soil:
    def __init__(self):

        df = pd.read_csv('./data/soils/all-data.csv')

        df.drop(columns=('Unnamed: 0'),inplace=True)

        df.rename(
            columns={
                'p': 'IndoorOutdoorPressure',
                'alpha': 'AttenuationGroundwater',
                'beta': 'RelativeAirEntryRate',
                'Pe': 'Peclet',
                'c': 'IndoorConcentration',
                'n': 'EntryRate',
                'soil_type': 'SoilType',
            },
            inplace=True,
        )

        df['SoilType']=df['SoilType'].map(lambda x: x.title().replace('-', ' '))
        # data choosing
        df['logIndoorConcentration'] = df['IndoorConcentration'].apply(np.log10)
        df['logAttenuationGroundwater'] = df['AttenuationGroundwater'].apply(np.log10)

        self.data = df

        return
