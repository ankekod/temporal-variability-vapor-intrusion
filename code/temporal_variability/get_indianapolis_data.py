import numpy as np
import pandas as pd
import sqlite3



db_dir = 'C:\\Users\\jstroem\\Dropbox\\var\\'

"""

indianapolis = pd.read_sql_query(
"SELECT C.StopDate, C.StopTime AS TimeStamp, C.Value AS Concentration, C.Variable as Specie FROM VOC_Data_SRI_8610_Onsite_GC C, Observation_Status_Data O WHERE DATE(C.StopDate)=DATE(O.StopDate) AND O.Value='not yet installed' AND (C.Location='422BaseS' OR C.Location='422BaseN');", db_indianapolis, )

indianapolis = indianapolis.assign(StopTime=lambda x: x['StopDate']+' '+x['TimeStamp'])
indianapolis.drop(columns=['StopDate','TimeStamp'],inplace=True)
indianapolis['StopTime'] = indianapolis['StopTime'].apply(pd.to_datetime)
indianapolis.sort_values(by=['StopTime','Specie'],inplace=True)
"""
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


class Indianapolis:

    def __init__(self):

        self.db = sqlite3.connect(get_dropbox_path() + '/var/Indianapolis.db')
        indoor = self.get_indoor_air()
        print(indoor)
        return


    def process_time(self,df):
        df = df.assign(Time=lambda x: x['StopDate']+' '+x['StopTime'])
        df.drop(columns=['StopDate','StopTime'],inplace=True)

        df['Time'] = df['Time'].apply(pd.to_datetime)
        df.sort_values(by=['Time'],inplace=True)
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
                'Variable': 'Specie',
                'Value': 'IndoorConcentration',
            },
            inplace=True,
        )

        indoor.drop(columns=['Depth_ft','Location'],inplace=True)
        indoor.set_index('Time',inplace=True)
        return indoor


ind = Indianapolis()
