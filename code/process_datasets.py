import numpy as np
import pandas as pd
import sqlite3

import os
dropbox_folder = None

def get_dropbox_path():
    for dirname, dirnames, filenames in os.walk(os.path.expanduser('~')):
        for subdirname in dirnames:
            if(subdirname == 'Dropbox'):
                dropbox_folder = os.path.join(dirname, subdirname)
                break
        if dropbox_folder:
            break
    return dropbox_folder

# TODO: create this function to intelligently find infs
def replace_inf(x):
    x = 1
    return

def get_season(x):
    seasons = {
        'Winter': (12, 2),
        'Spring': (3, 5),
        'Summer': (6, 8),
        'Fall': (9, 11),
    }
    if (x == 12) or (x == 1) or (x == 2):
        return 'Winter'
    elif (x == 3) or (x == 4) or (x == 5):
        return 'Spring'
    elif (x == 6) or (x == 7) or (x == 8):
        return 'Summer'
    elif (x == 9) or (x == 10) or (x == 11):
        return 'Fall'
    else:
        return 'Error'

class Indianapolis:
    def __init__(self):

        self.db = sqlite3.connect(get_dropbox_path() + '/var/Indianapolis.db')
        df = self.get_data()
        df.to_csv('./data/indianapolis.csv')
        return

    def get_data(self):
        subslab = self.get_subslab()
        indoor = self.get_indoor_air()
        ssd = self.get_ssd_status()

        df = pd.merge_asof(indoor, subslab, left_index=True, right_index=True, by='Specie')
        df = df.loc[(df.index.date>=ssd.index.date.min()) & (df.index.date<=ssd.index.date.max())]

        df['AttenuationSubslab'] = df['IndoorConcentration']/df['SubslabConcentration']


        df['AttenuationSubslab'] = df['AttenuationSubslab'].replace([-np.inf,np.inf], np.nan)
        df['logIndoorConcentration'] = df['IndoorConcentration'].apply(np.log10).replace([-np.inf,np.inf], np.nan)
        df['logAttenuationSubslab'] = df['AttenuationSubslab'].apply(np.log10).replace([-np.inf,np.inf], np.nan)



        pressure = self.get_pressure()

        df = pd.merge_asof(df, pressure, left_index=True, right_index=True,)

        df['Season'] = df.index.month.map(get_season)

        return df

    def get_pressure(self):
        pressure = pd.read_sql_query(
            "SELECT StopDate, StopTime, Variable, Value, Location FROM Differential_Pressure_Data;",
            self.db,
        )
        pressure = self.process_time(pressure)
        pressure = pressure.loc[(pressure['Location']=='422') & (pressure['Variable']=='Basement.Vs.Exterior')]
        pressure.rename(
            columns={
                'Value': 'IndoorOutdoorPressure',
            },
            inplace=True,
        )
        #pressure['IndoorOutdoorPressure'] *= -1 # changing sign to my convention (wasn't it needed?)
        pressure.drop(columns=['Variable','Location'],inplace=True)
        return pressure

    def get_ssd_status(self):

        ssd = pd.read_sql_query(
            "SELECT StopDate, StopTime, Variable, Value FROM Observation_Status_Data;",
            self.db,
        )

        ssd = self.process_time(ssd)
        ssd = ssd.loc[(ssd['Variable']=='Mitigation') & (ssd['Value']=='not yet installed')]
        return ssd

    def process_time(self,df):
        df = df.assign(Time=lambda x: x['StopDate']+' '+x['StopTime'])
        df.drop(columns=['StopDate','StopTime'],inplace=True)

        df['Time'] = df['Time'].apply(pd.to_datetime)
        df.sort_values(by=['Time'],inplace=True)
        df.set_index('Time',inplace=True)

        return df

    # retrieves the indoor air concentration in 422BaseS or ...N
    def get_indoor_air(self):
        indoor = pd.read_sql_query(
            "SELECT StopDate, StopTime, Variable, Value, Location, Depth_ft FROM VOC_Data_SRI_8610_Onsite_GC;",
            self.db,
        )
        indoor = self.process_time(indoor)
        indoor = indoor.loc[(indoor['Location']=='422BaseN') | (indoor['Location']=='422BaseS')]
        indoor.rename(
            columns={
                #'Variable': 'IndoorSpecie',
                'Variable': 'Specie',
                'Value': 'IndoorConcentration',
            },
            inplace=True,
        )
        indoor.drop(columns=['Depth_ft','Location'],inplace=True)
        return indoor

    def get_subslab(self):
        subslab = pd.read_sql_query(
            "SELECT StopDate, StopTime, Variable, Value, Location, Depth_ft FROM VOC_Data_SRI_8610_Onsite_GC;",
            self.db,
        )
        subslab = self.process_time(subslab)
        subslab = subslab.loc[(subslab['Location']=='SSP-4')]
        subslab.rename(
            columns={
                #'Variable': 'SubslabSpecie',
                'Variable': 'Specie',
                'Value': 'SubslabConcentration',
            },
            inplace=True,
        )
        subslab.drop(columns=['Depth_ft','Location'],inplace=True)
        return subslab

class ASUHouse:
    def __init__(self):

        self.db = sqlite3.connect(get_dropbox_path() + '/var/HillAFB.db')
        df = self.get_data()
        print(df)
        df.to_csv('./data/asu_house.csv')
        return

    def get_data(self):
        subslab = self.get_subslab()
        indoor = self.get_indoor_air()
        phases = self.get_phases()
        pressure = self.get_pressure()
        ae = self.get_air_exchange()
        avg_gw = self.get_avg_groundwater()

        df = pd.merge_asof(indoor, subslab, left_index=True, right_index=True)
        df = pd.merge_asof(df, avg_gw, left_index=True, right_index=True,)

        df['AttenuationSubslab'] = df['IndoorConcentration']/df['SubslabConcentration']
        df['AttenuationAvgGroundwater'] = df['IndoorConcentration']/df['AvgGroundwaterConcentration']



        df['AttenuationSubslab'] = df['AttenuationSubslab'].replace([-np.inf,np.inf], np.nan)
        df['logIndoorConcentration'] = df['IndoorConcentration'].apply(np.log10).replace([-np.inf,np.inf], np.nan)
        df['logAttenuationSubslab'] = df['AttenuationSubslab'].apply(np.log10).replace([-np.inf,np.inf], np.nan)
        df['logAttenuationAvgGroundwater'] = df['AttenuationAvgGroundwater'].apply(np.log10).replace([-np.inf,np.inf], np.nan)


        df = pd.merge_asof(df, pressure, left_index=True, right_index=True,)
        df = pd.merge_asof(df, ae, left_index=True, right_index=True,)

        pp_status = lambda x: 'Open' if x <= phases.index.values[0] else ('Closed' if x >= phases.index.values[-2] else 'CPM')
        df['Phase'] = df.index.map(pp_status)
        df['Season'] = df.index.month.map(get_season)

        return df

    def get_avg_groundwater(self):
        avg_gw = pd.read_sql_query(
            "SELECT StopTime, Concentration AS AvgGroundwaterConcentration FROM AverageGroundwaterConcentration;",
            self.db,
        )
        avg_gw['AvgGroundwaterConcentration'] *= 1e3 # converts from ug/L to ug/m^3
        avg_gw = self.process_time(avg_gw)
        return avg_gw

    def get_air_exchange(self):
        ae = pd.read_sql_query(
            "SELECT StopTime, AirExchangeRate FROM Tracer;",
            self.db,
        )

        ae = self.process_time(ae)
        return ae

    def get_pressure(self):
        pressure = pd.read_sql_query(
            "SELECT StopTime, Pressure AS IndoorOutdoorPressure, PressureSubslab AS SubslabPressure FROM PressureDifference;",
            self.db,
        )
        pressure = self.process_time(pressure)


        pressure['IndoorOutdoorPressure'] *= -1
        pressure['SubslabPressure'] *= -1

        return pressure

    def get_phases(self):

        phases = pd.read_sql_query(
            "SELECT StopTime AS Time FROM Phases;",
            self.db,
        )
        phases = self.process_time(phases)
        return phases

    def process_time(self,df):
        df.rename(columns={'StopTime': 'Time'},inplace=True)

        df['Time'] = df['Time'].apply(pd.to_datetime)
        df.sort_values(by=['Time'],inplace=True)
        df.set_index('Time',inplace=True)

        return df

    # retrieves the indoor air concentration in 422BaseS or ...N
    def get_indoor_air(self):
        indoor = pd.read_sql_query(
            "SELECT StopTime, Concentration AS IndoorConcentration FROM TDBasement;",
            self.db,
        )
        indoor = self.process_time(indoor)

        return indoor

    def get_subslab(self):
        subslab = pd.read_sql_query(
            "SELECT StopTime, Concentration AS SubslabConcentration, Location, Depth FROM SoilGasConcentration;",
            self.db,
        )
        subslab = self.process_time(subslab)
        subslab = subslab.loc[(subslab['Depth']==0.0) & (subslab['Location']=='6')]
        subslab.drop(columns=['Depth','Location'],inplace=True)
        return subslab


class NorthIsland:
    def __init__(self):

        self.db = sqlite3.connect(get_dropbox_path() + '/var/NorthIslandNAS.db')
        df = self.get_data()
        df.to_csv('./data/north_island.csv')
        return

    def get_data(self):
        indoor = self.get_indoor_air()
        pressure = self.get_pressure()

        df = pd.merge_asof(indoor, pressure, on='Time')

        return df

    def get_indoor_air(self):
        indoor = pd.read_sql_query(
            "SELECT Time, Concentration AS IndoorConcentration FROM IndoorAirConcentration;",
            self.db,
        ).sort_values(by='Time')

        indoor['logIndoorConcentration'] = indoor['IndoorConcentration'].apply(np.log10).replace([-np.inf,np.inf], np.nan)
        return indoor

    def get_pressure(self):
        pressure = pd.read_sql_query(
            "SELECT Time, Pressure AS IndoorOutdoorPressure FROM IndoorOutdoorPressureDifference;",
            self.db,
        ).sort_values(by='Time')
        pressure['IndoorOutdoorPressure'] *= -1
        return pressure

#ind = Indianapolis()
asu = ASUHouse()
#nas = NorthIsland()
