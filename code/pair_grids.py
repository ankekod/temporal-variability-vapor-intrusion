import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats



asu = pd.read_csv('./data/asu_house.csv')
indianapolis = pd.read_csv('./data/indianapolis.csv')

# data processing
asu = asu.loc[(asu['IndoorOutdoorPressure'] >= -5) & (asu['IndoorOutdoorPressure'] <= 5)] # TODO: kind of ugly "hack", should instead adjust axes

# global settings
num_bins = 10
fig_path = './figures/pair_grids/'
ext = '.png'
dpi = 300

# method to set custom axis labels
def custom_axis(g):

    replacements = {
        'IndoorOutdoorPressure': '$\\Delta p_\\mathrm{in/out}$ (Pa)',
        'AirExchangeRate': '$A_e$ (1/h)',
        'logIndoorConcentration': '$\\log{(c_\\mathrm{in})}$ ($\\mathrm{\\mu g/m^3}$)',
        'logAttenuationSubslab': '$\\log{(\\alpha_\\mathrm{subslab})}$'
    }

    for i in range(len(g.axes)):
        for j in range(len(g.axes)):
            xlabel = g.axes[i][j].get_xlabel()
            ylabel = g.axes[i][j].get_ylabel()
            if xlabel in replacements.keys():
                g.axes[i][j].set_xlabel(replacements[xlabel])
            if ylabel in replacements.keys():
                g.axes[i][j].set_ylabel(replacements[ylabel])
    return g


# demonstrate the variability

# asu
# PP hue
g = sns.PairGrid(
    asu.loc[asu['Phase']!='CPM'][['IndoorOutdoorPressure','AirExchangeRate','logIndoorConcentration','logAttenuationSubslab','Phase']],
    hue='Phase',
)
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.regplot, x_bins=num_bins, truncate=True, )
g = g.map_diag(sns.kdeplot, shade=True)
g = g.add_legend()
g = custom_axis(g)
plt.savefig(fig_path+'asu_phase'+ext,dpi=dpi)

# season hue
g = sns.PairGrid(
    asu.loc[asu['Phase']=='Closed'][['IndoorOutdoorPressure','AirExchangeRate','logIndoorConcentration','logAttenuationSubslab','Season']],
    hue='Season',
    hue_order=['Winter', 'Fall', 'Spring', 'Summer'],
)
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.regplot, x_bins=num_bins, truncate=True, )
g = g.map_diag(sns.kdeplot, shade=True)
g = g.add_legend()
g = custom_axis(g)
plt.savefig(fig_path+'asu_closed_season'+ext,dpi=dpi)

# indianapolis
# specie hue
g = sns.PairGrid(
    indianapolis[['IndoorOutdoorPressure','logIndoorConcentration','logAttenuationSubslab','Specie']],
    hue='Specie',
)
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.regplot, x_bins=num_bins, truncate=True, )
g = g.map_diag(sns.kdeplot, shade=True)
g = g.add_legend()
g = custom_axis(g)
plt.savefig(fig_path+'indianapolis_species'+ext,dpi=dpi)

# season hue
g = sns.PairGrid(
    indianapolis.loc[indianapolis['Specie']=='Tetrachloroethene'][['IndoorOutdoorPressure','logIndoorConcentration','logAttenuationSubslab','Season']],
    hue='Season',
    hue_order=['Winter', 'Fall', 'Summer'],
)
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.regplot, x_bins=num_bins, truncate=True, )
g = g.map_diag(sns.kdeplot, shade=True)
g = g.add_legend()
g = custom_axis(g)
plt.savefig(fig_path+'indianapolis_pce_season'+ext,dpi=dpi)

#plt.show()
