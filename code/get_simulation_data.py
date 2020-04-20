import numpy as np
import pandas as pd

# TODO: Create class with common functions

def get_simulation_data():
    dfs = []
    for file in ('pp','pp_uniform','no_pp','pp_uncontaminated',):
        df = pd.read_csv('../data/simulation/sweep_'+file+'.csv',header=4)

        if file == 'pp':
            df['Pathway'] = np.repeat('Yes', len(df))
            df['Contaminated'] = np.repeat('Yes', len(df))
            df['Gravel'] = np.repeat('Yes', len(df))
        elif file == 'pp_uniform':
            df['Pathway'] = np.repeat('Yes', len(df))
            df['Contaminated'] = np.repeat('Yes', len(df))
            df['Gravel'] = np.repeat('No', len(df))
        elif file == 'no_pp':
            df['Pathway'] = np.repeat('No', len(df))
            df['Contaminated'] = np.repeat('No', len(df))
            df['Gravel'] = np.repeat('Yes', len(df))
        elif file == 'pp_uncontaminated':
            df['Pathway'] = np.repeat('Yes', len(df))
            df['Contaminated'] = np.repeat('No', len(df))
            df['Gravel'] = np.repeat('Yes', len(df))

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
            #'Average subslab concentration, Global Evaluation: Average TCE in subslab {gev10}': 'SubslabConcentration',
        },
        inplace=True,
    )

    print(df['SubslabConcentration'])
    # data choosing
    df['AirExchangeRate'] *= 3600 # convert from 1/s to 1/hr
    df['AirExchangeRate'] = np.around(df['AirExchangeRate'], 1)
    df['logIndoorConcentration'] = df['IndoorConcentration'].apply(np.log10)
    df['logAttenuationSubslab'] = df['AttenuationSubslab'].apply(np.log10)
    df['logAttenuationGroundwater'] = df['AttenuationGroundwater'].apply(np.log10)
    df['logSubslabConcentration'] = df['SubslabConcentration'].apply(np.log10)

    return df


class PreferentialPathway:
    def __init__(self):
        dfs = []
        for file in ('pp','pp_uniform','no_pp','pp_uncontaminated',):
            df = pd.read_csv('../data/simulation/sweep_'+file+'.csv',header=4)
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
        df['AirExchangeRate'] = np.around(df['AirExchangeRate'], 1)
        df['logIndoorConcentration'] = df['IndoorConcentration'].apply(np.log10)
        df['logAttenuationSubslab'] = df['AttenuationSubslab'].apply(np.log10)
        df['logAttenuationGroundwater'] = df['AttenuationGroundwater'].apply(np.log10)

        self.data = df
        return

    def get_data(self):
        return self.data

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

def get_diurnal_simulation_data():
    dfs = []
    for file in ('closed','closed_const_ae','open','open_const_ae',):
        df = pd.read_csv('../data/diurnal/simulation_results/pp_'+file+'.csv',header=4)

        if 'closed' in file:
            df['Pathway'] = np.repeat('No', len(df))
        elif 'open' in file:
            df['Pathway'] = np.repeat('Yes', len(df))


        if 'const' in file:
            df['ConstAe'] = np.repeat('Yes', len(df))
        elif 'const' not in file:
            df['ConstAe'] = np.repeat('No', len(df))

        dfs.append(df)

    df = pd.concat(dfs,sort=True)
    df.rename(
        columns={
            '% Time (h)': 'Time',
            'IndoorOutdoorPressure (Pa)': 'IndoorOutdoorPressure',
            'IndoorConcentration (ug/m^3)': 'IndoorConcentration',
            'ExitRate (ug/hour)': 'ExitRate',
            'EntryRate (ug/hour)': 'EntryRate',
            'AttenuationGroundwater (1)': 'AttenuationGroundwater',
            'AirExchangeRate (1/hr)': 'AirExchangeRate'
            },
        inplace=True,
    )


    # data choosing
    #df['AirExchangeRate'] *= 3600 # convert from 1/s to 1/hr
    #df['AirExchangeRate'] = np.around(df['AirExchangeRate'], 1)
    df['logIndoorConcentration'] = df['IndoorConcentration'].apply(np.log10)
    #df['logAttenuationSubslab'] = df['AttenuationSubslab'].apply(np.log10)
    df['logAttenuationGroundwater'] = df['AttenuationGroundwater'].apply(np.log10)
    #df['logSubslabConcentration'] = df['SubslabConcentration'].apply(np.log10)

    return df

df = get_diurnal_simulation_data()
df.to_csv('diurnal.csv', index=False)
print(df)
#df = get_simulation_data()
#df.to_csv('asu_house_simulation_data.csv', index=False)
