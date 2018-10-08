import numpy as np
import pandas as pd
import pandas.tools.plotting as pdplt
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import sqlite3
data_dir = './data/preferential-pathway-sensitivity/'
db_dir = '/home/jonathan/lib/vapor-intrusion-dbs/'

"""
Prepares model data
"""
# no PP data
no_pp = pd.read_csv(
    data_dir + 'param-no-preferential-pathway.csv',
    header=4,
)
no_pp['PP'] = pd.Series(np.repeat('No', len(no_pp['p'])))
# PP data
pp = pd.read_csv(
    data_dir + 'param-preferential-pathway.csv',
    header=4,
)
pp['PP'] = pd.Series(np.repeat('Yes', len(no_pp['p'])))
df = pd.concat([pp, no_pp],axis=0,sort=True)
# adds friendlier tags for gravel sub-base present (SB) and if the PP is
# contaminated (chi)
df['Gravel sub-base'] = pd.Series(df['SB'].apply(lambda x: 'Yes' if x == 1 else 'No'))
df['Contaminated PP'] = pd.Series(df['chi'].apply(lambda x: 'Yes' if x == 1 else 'No'))
# data from permeable model, unknown if I want to add it,
# might be able to concat at it and assign SB = Sand or whatever
permamble = pd.read_csv(
    data_dir + 'param-permeable-soil.csv',
    header=4,
)
"""
Prepares ASU data for comparison 
"""
db_asu = sqlite3.connect(db_dir + 'asu_house.db')
asu = pd.read_sql_query(
    "SELECT \
        day, \
        pressure_difference AS Pressure, \
        tce_emission_rate, \
        building_flow_rate, tce_groundwater \
    FROM \
        parameters; \
    ", db_asu,
).interpolate(limit_area='inside',limit=250).dropna()
# removes data that is outside model data range
asu['Pressure'] = -1.0*asu['Pressure']
asu = asu[(asu['Pressure'] < 10) & (asu['Pressure'] > -10)]
# removes cpm period from dataset
cpm_start, cpm_end = 780.0, 1157.0
asu = asu[(asu['day'] < cpm_start) | (asu['day'] > cpm_end)]
# assigns a PP open/closed column (for sorting purposes)
asu['PP'] = pd.Series(asu['day'].apply(lambda x: 'Open' if x < cpm_start else 'Closed'))
# calculates attentuation factor
asu['Concentration'] = asu['tce_emission_rate']/asu['building_flow_rate']
asu['Attenuation factor'] = asu['Concentration']/asu['tce_groundwater']

for df_tag, asu_tag in zip(['Yes','No'],['Open','Closed']):
    fix, ax = plt.subplots()

    if df_tag == 'No':
        df = df.drop(no_pp[no_pp.chi == 0].index)
        pivot_cols = 'Gravel sub-base'
        legend_title = 'Gravel sub-base?'
    else:
        pivot_cols = ['Gravel sub-base', 'Contaminated PP']
        legend_title = '(Gravel sub-base?, Contaminant in PP?)'


    df[df.PP == df_tag].pivot_table(index='p', columns=pivot_cols, values='alpha').plot(
        ax=ax,
        legend=True,
        linewidth=2.5,
    )

    sns.regplot(
        ax=ax,
        data=asu[asu.PP == asu_tag],
        x='Pressure',
        y='Attenuation factor',
        x_bins=np.linspace(-10,10,40),
        label='ASU data, PP %s' % asu_tag.lower()
    )

    ax.set(yscale="log")

    ax.set_xlabel('Indoor/outdoor pressure difference (Pa)')
    ax.set_ylabel('Attentuation factor (vapor source)')
    ax.set_ylim([2e-7,5e-3])

    ax.legend(
        title = legend_title,
        loc='best',
    )
    fig_dir = './figures/preferential-pathway-sensitivity/'
    plt.savefig(fig_dir + '%s-pp' % df_tag.lower() + '.pdf', dpi=300)
    plt.show()
