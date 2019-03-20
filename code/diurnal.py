import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy import stats


class Diurnal:
    def __init__(self):
        data = pd.read_csv('./data/asu_house.csv')
        data = data.loc[data['Phase']!='CPM']
        data['Time'] = data['Time'].apply(pd.to_datetime)


        pressure = data[['Time','IndoorOutdoorPressure']].groupby(data['Time'].dt.hour).median()
        air_exchange_rate = data[['Time','AirExchangeRate']].groupby(data['Time'].dt.hour).median()

        pressure.to_csv('./data/diurnal/pressure.csv')
        air_exchange_rate.to_csv('./data/diurnal/air_exchange_rate.csv')

        return


Diurnal()
