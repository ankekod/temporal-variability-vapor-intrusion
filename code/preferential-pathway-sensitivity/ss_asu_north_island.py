import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
from scipy import stats
from natu import units

# path where the datafiles are stored
file_path = '/Users/duckerfeller/Dropbox/Research/Data/'


asu = pd.read_csv(
    file_path+'data_asu2.csv', # full path
    header=4, # dataframe headers are in row 4
)

nas = pd.read_csv(
    file_path+'data_north_island.csv', # full path
    header=4, # dataframe headers are in row 4
)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(7,4),sharey=True,sharex=True)

asu = asu.pivot_table(
    index=['p'], # x-axis
    values='alpha', # y-axis
    columns=['Ae'], # levels
)

titles = (
    'No PP\nGravel sub-base\nSmaller foundation crack area',
    'No PP\nSand soil',
)

fig, ax = plt.subplots(figsize=(5,3))

# plots the pivoted data
asu.plot(
    ax=ax,
    logy=True, # sets y-axis to log scale
    color='k', # sets the line color to black
    #title=titles[0], # assigns the custom titles
    style=['-','--'],
    legend=False, # removes the default legend (replaced by titeles)
)

ax.set_xlabel('Indoor/outdoor pressure\ndifference (Pa)')
ax.set_ylabel('Attenuation factor')
ax.set_title(titles[0])
ax.legend(['Ae = 0.1 (1/h)', 'Ae = 0.5 (1/h)'],loc='best')
plt.tight_layout()
plt.savefig(
    '/Users/duckerfeller/Documents/Research/Transient Variability/Figures/ss_asu.png',
    dpi=300,
)




fig, ax = plt.subplots(figsize=(5,3))
# plots the pivoted data
nas.plot(
    x='p',
    y='alpha',
    ax=ax,
    logy=True, # sets y-axis to log scale
    subplots=True, # indicates we want to plot each panda level as a subplot
    color='k', # sets the line color to black
    #title=titles[1], # assigns the custom titles
    legend=False, # removes the default legend (replaced by titeles)
)

ax.set_xlabel('Indoor/outdoor pressure\ndifference (Pa)')
ax.set_ylabel('Attenuation factor')
ax.set_title(titles[1])
plt.tight_layout()

plt.savefig(
    '/Users/duckerfeller/Documents/Research/Transient Variability/Figures/ss_north_island.png',
    dpi=300,
)
#plt.show()
