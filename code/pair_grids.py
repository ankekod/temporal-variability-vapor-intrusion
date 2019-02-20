import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats



asu = pd.read_csv('./data/asu_house.csv')
indianapolis = pd.read_csv('./data/indianapolis.csv')


# global settings
num_bins = 5


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


print(indianapolis.loc[indianapolis['Specie']=='Tetrachloroethene']['Season'].value_counts())

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

plt.show()
