import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy import stats
from get_simulation_data import Soil


# TODO: Run new transient simulation with NAS pressure data?
class Figure1:
    def __init__(self):
        data = pd.read_csv('./data/north_island.csv').dropna()
        sim = Soil().data
        sim = sim.loc[(sim['SoilType']=='Sand')]


        ax = sns.kdeplot(
            data=data['IndoorOutdoorPressure'],
            data2=data['logIndoorConcentration']-data['logIndoorConcentration'].mean(),
            shade_lowest=False,
            shade=True,
        )
        # TODO: Normalise to when Pa = 0
        sim['logAttenuationGroundwaterFromMean'] = sim['logAttenuationGroundwater'] - sim.loc[sim['IndoorOutdoorPressure']==-1]['logAttenuationGroundwater'].values
        ax = sns.lineplot(
            data=sim,
            x='IndoorOutdoorPressure',
            y='logAttenuationGroundwaterFromMean',
            ax=ax,
        )

        ax.set(
            xlim=[-30,15],
        )

        plt.show()

        return


Figure1()
