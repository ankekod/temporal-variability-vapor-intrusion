import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
from scipy import stats
from natu import units

# path where the datafiles are stored
file_path = '/Users/duckerfeller/Dropbox/Research/Data/'

# names of the data files we wish to load
file_names = (
    'data_pp.csv',
    'data_no_pp.csv',
)


data_list = [] # list to store dataframes
for file_name in file_names: # loops through files
    # loads dataframe based on file
    data = pd.read_csv(
        file_path+file_name, # full path
        header=4, # dataframe headers are in row 4
    )
    # assigns if there is a PP in the dataset based on the naming of the file
    data = data.assign(
        PP = np.repeat(
            'No' if 'no_pp' in file_name else 'Yes',
            data.shape[0],
        )
    )
    data_list.append(data) # appends dataframe to list

# concatenates the dataframes
df = pd.concat(
    data_list, # dataframes
    axis=0, # should be added row-wise (on top of each other)
    ignore_index=True, # do not concat according to index
)

# removes the case where there is no PP and the PP isn't contaminated
# (it is superfluous to have two no PP cases...)
df = df[(df.chi != 0.0) | (df.PP != 'No')]

# pivots the dataframe to 3 "levels" which are chi, PP and SB
df = df.pivot_table(
    index='p', # x-axis
    values='alpha', # y-axis
    columns=['chi', 'PP', 'SB'], # levels
)

# all of the titles to use
titles = (
    'Uncontamianted PP\nGravel sub-base',
    'Uncontaminated PP\nSandy clay sub-base',
    'No PP\nGravel sub-base',
    'No PP\nSandy clay sub-base',
    'Contaminated PP\nGravel sub-base',
    'Contaminated PP\nSandy clay sub-base',
)

cols = [series for colname, series in df.iteritems()]




for i, col in enumerate(cols):
    fig, ax = plt.subplots(figsize=(5,3))
    col.plot(
        ax=ax,
        logy=True,
        legend=False,
        color='k',
    )
    ax.set_ylim([1e-07, 5e-03])
    ax.set_title(titles[i])

    ax.set_ylabel('Attenuation factor')
    ax.set_xlabel('Indoor/outdoor pressure difference (Pa)')
    plt.tight_layout()
    plt.savefig(
        '../Figures/ss_pp_sensitivity_analysis_%i.png' % i,
        dpi=300,
    )
    #plt.show()
"""
# plots the pivoted data
axes = df.plot(
    layout=(3,2), # number of subplots 3x2 rxc
    figsize=(7,7), # figure size
    sharex=True, # all subplots share x axises
    sharey=True, # - " - y
    logy=True, # sets y-axis to log scale
    #subplots=True, # indicates we want to plot each panda level as a subplot
    color='k', # sets the line color to black
    #title=titles, # assigns the custom titles
    legend=False, # removes the default legend (replaced by titeles)
)

# sets axis labels for each subplot
for ax_row in axes: # loops through each subplot row
    for ax in ax_row: # loops through each subplot column in the row
        ax.set_xlabel('Indoor/outdoor pressure\ndifference (Pa)')
        ax.set_ylabel('Attenuation factor')
"""
# formating
plt.tight_layout()
"""
plt.savefig(
    '/Users/duckerfeller/Documents/Research/Transient Variability/Figures/ss_pp_sensitivity_analysis.png',
    dpi=300,
)
"""
#plt.show()
