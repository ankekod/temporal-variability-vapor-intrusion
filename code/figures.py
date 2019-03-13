import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy import stats
from get_simulation_data import Soil, PreferentialPathway


"""
This figure shows a case (North Island NAS) where indoor/outdoor pressure
difference the key driver at a VI site, and how are are able to model this
by assuming a permeable soil type (sand).
"""
class Figure1:
    def __init__(self,y_data_log=False , norm_conc=True):
        nas = pd.read_csv('./data/north_island.csv').dropna()
        asu = pd.read_csv('./data/asu_house.csv').dropna()
        asu_open = asu.loc[asu['Phase']=='Open']
        asu_closed = asu.loc[asu['Phase']=='Closed']


        x='IndoorOutdoorPressure'

        if y_data_log is True:
            y='logIndoorConcentration'
        else:
            y='IndoorConcentration'


        datasets = (nas, asu_open, asu_closed,)
        labels = ('North Island NAS', 'ASU house, PP open', 'ASU house, PP closed',)

        fig, ax = plt.subplots()

        for data, label in zip(datasets, labels):
            if norm_conc is True:
                if y_data_log is True:
                    data2 = data[y]-data[y].mean()
                else:
                    data2 = data[y]/data[y].mean()

            else:
                data2 = data[y]


            r, p = stats.pearsonr(data[x], data2)

            print(label)
            print(data[x].describe(percentiles=[0.05,0.95]), data['IndoorConcentration'].describe(percentiles=[0.05,0.95]), data2.describe(percentiles=[0.05,0.95]))
            sns.kdeplot(
                data=data[x],
                data2=data2,
                shade_lowest=False,
                shade=True,
                label=label + ', r = %1.2f' % r, # TODO: Add Pearson's r analysis
                ax=ax,
            )



        yticks, yticklabels = get_log_ticks(-1, 1.5)
        print(yticklabels)
        # formatting options
        ax.set(
            xlim=[-30,15],
            ylim=[-1,1.5],
            xlabel='$p_\\mathrm{in/out} \\; \\mathrm{(Pa)}$',
            ylabel='$c_\\mathrm{in}/c_\\mathrm{in,mean}$',
            title='Relationship between indoor/outdoor pressure difference and\nTCE in indoor air (normalized to dataset mean concentration)',
            #yscale='log',
            yticks=yticks,
            yticklabels=yticklabels,
        )
        plt.legend(loc='upper left')
        #plt.savefig('./figures/2d_kde/nas_asu_pp.pdf', dpi=300)
        #plt.savefig('./figures/2d_kde/nas_asu_pp.png', dpi=300)

        plt.show()

        return


def get_log_ticks(start, stop, style='f'): # TODO: Remove the unnecessary ticklabels

    ticks = np.array([])
    for int_now in np.arange(np.floor(start),np.ceil(stop)+1):
        ticks = np.append(ticks, np.arange(0.1,1.1,0.1)*10.0**int_now)


    if style == 'f':
        labels = ['%1.1f' % tick for tick in ticks]
    elif style=='e':
        labels = ['%1.1e' % tick for tick in ticks]
    ticks = np.log10(ticks)


    return ticks, labels


class AttenuationSubslab:
    def __init__(self):
        asu = pd.read_csv('./data/asu_house.csv')
        asu = asu.loc[asu['Phase']!='CPM']

        ax = sns.boxplot(x="Phase", y="logAttenuationSubslab", data=asu)

        ticks, labels = get_log_ticks(-4,2, style='e')
        ax.set(
            xlabel='Preferential pathway status',
            ylabel='Attenuation from subslab',
            #yticks=ticks,
            #yticklabels=labels,
        )

        plt.tight_layout()
        plt.savefig('./figures/temporal_variability/asu_attenuation_subslab.pdf', dpi=300)
        plt.savefig('./figures/temporal_variability/asu_attenuation_subslab.png', dpi=300)

        plt.show()
        return

#Figure1(y_data_log=True,norm_conc=True)
AttenuationSubslab()
