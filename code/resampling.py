import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import LSODA
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import sqlite3
import seaborn as sns


# global settings
num_bins = 10
fig_path = './figures/pair_grids/'
ext = '.png'
dpi = 300


asu = pd.read_csv('./data/asu_house.csv')
indianapolis = pd.read_csv('./data/indianapolis.csv')


resampled = pd.DataFrame({'Resampling': [], 'MaxDelta': [], 'Dataset': []})

datasets = (
    asu[asu['Phase']=='Open'],
    asu[asu['Phase']=='Closed'],
    indianapolis[indianapolis['Specie']=='Trichloroethene'],
    indianapolis[indianapolis['Specie']=='Tetrachloroethene'],
    indianapolis[indianapolis['Specie']=='Chloroform'],
)
names = (
    'ASU House, PP Open',
    'ASU House, PP Closed',
    'Indianapolis, TCE',
    'Indianapolis, PCE',
    'Indianapolis, Chloroform',
)

for dataset, name in zip(datasets,names):
    dataset['Time'] = dataset['Time'].apply(pd.to_datetime)
    print('Using dataset: %s' % name)
    for resampling_time in ('1D','2D','3D','1W','2W','3W','1M','2M','3M','6M',):
        r = dataset.resample(resampling_time ,on='Time', kind='timestamp')
        r = r['logIndoorConcentration'].agg([np.mean, np.max, np.min, np.std])
        to_be_appended = pd.DataFrame({
            'Resampling': np.repeat(resampling_time,len(r)),
            'MaxDelta': r['amax'].values-r['amin'].values,
            'Dataset': np.repeat(name,len(r)),
        })
        resampled = resampled.append(to_be_appended,ignore_index=True)

resampled.dropna(inplace=True)

#print(resampled)
sns.set_palette(sns.color_palette("muted"))


g = sns.catplot(x="Resampling", y="MaxDelta", hue='Dataset', kind="point", data=resampled, legend_out=False, aspect=1.5)

my_ytick_labels = ["%1.1f" % y_tick for y_tick in 10.0**g.ax.get_yticks()]
g.ax.set_yticklabels(my_ytick_labels)
g.ax.set_xlabel('Period')
g.ax.set_ylabel('$c_\\mathrm{max}/c_\\mathrm{min}$')
g.ax.set_title('Maximum variability within a given period')

#plt.show()
plt.savefig('./figures/temporal_variability/resampling.png',dpi=300)
