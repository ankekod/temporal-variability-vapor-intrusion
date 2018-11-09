import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import LSODA
from scipy.interpolate import interp1d
import sqlite3
import seaborn as sns

data_dir = './data/preferential-pathway-sensitivity/'
db_dir = '/home/jonathan/lib/vapor-intrusion-dbs/'
#db_dir = 'C://Users/jstroem/lib/vapor-intrusion-dbs/'

db = sqlite3.connect(db_dir + 'hill-afb.db')

asu = pd.read_sql_query(
    "SELECT\
        da.*, \
        gw.concentration as gw_concentration \
    FROM daily_averages da\
    LEFT JOIN groundwater_concentration gw USING(time)\
    UNION ALL\
    SELECT\
        da.*, \
        gw.concentration as gw_concentration \
    FROM groundwater_concentration gw\
    LEFT JOIN daily_averages da USING(time)\
    WHERE da.time IS NULL;",
    db,
).interpolate(method='piecewise_polynomial')

K_H = 0.403
asu['alpha'] = asu['concentration']/(asu['gw_concentration']*1e3*K_H)


phases = pd.read_sql_query("SELECT * from phases;", db)
asu.air_exchange_rate *= 24.0
asu.time = asu.time.apply(pd.to_datetime)
phases.start = phases.start.apply(pd.to_datetime)
phase1 = phases[(phases.cpm == 'off') & (phases.land_drain == 'open')]
phase3 = phases[(phases.cpm == 'off') & (phases.land_drain == 'closed')]
filter1 = (asu['time'] < phase1['stop'].values[0])
filter3 = (asu['time'] > phase3['start'].values[0])

pre_cpm = asu.loc[filter1].copy()
post_cpm = asu.loc[filter3].copy()

convert_time = lambda df: (df.time.diff()/np.timedelta64(1,'D')).apply(float)

pre_cpm['dt'] = convert_time(pre_cpm)
post_cpm['dt'] = convert_time(post_cpm)

def dc(df,log=True,type='concentration'):
    if log == True:
        df[type] = df[type].apply(np.log10)
    return df[type].diff()

pre_cpm['dc'] = dc(pre_cpm,log=True)
pre_cpm['dcdt'] = pre_cpm['dc']/pre_cpm['dt']

post_cpm['dc'] = dc(post_cpm,log=True)
post_cpm['dcdt'] = post_cpm['dc']/post_cpm['dt']

pre_cpm['dalpha'] = dc(pre_cpm,log=True,type='alpha')
pre_cpm['dalphadt'] = pre_cpm['dalpha']/pre_cpm['dt']

post_cpm['dalpha'] = dc(post_cpm,log=True,type='alpha')
post_cpm['dalphadt'] = post_cpm['dalpha']/post_cpm['dt']


df = pd.read_csv('./data/transient-response/cstr-changes.csv')
mol_m3_to_ug_m3 = 131.4*1e6
df[['c_max','c_min']] *= mol_m3_to_ug_m3
df[['c_max','c_min']] = df[['c_max','c_min']].apply(np.log10)
df[['t_down','t_up']] /= 24.0
df['dcdt_up'] = (df['c_max']-df['c_min'])/df['t_up']
df['dcdt_down'] = (df['c_min']-df['c_max'])/df['t_down']

df = df[df['Ae']==0.5]



fig, ax = plt.subplots()
sns.kdeplot(pre_cpm['dcdt'],ax=ax,label='c, PP open')
sns.kdeplot(post_cpm['dcdt'],ax=ax,label='c, PP closed')
ax.set_xlabel('$\\frac{\\Delta \\log(c_\\mathrm{indoor})}{\\Delta t}$')

ys = np.linspace(0.5,3.0,len(df.index))

print(ys)
for i, soil, dcdt_down, dcdt_up in zip(df.index, df.soil, df.dcdt_down, df.dcdt_up):
    ax.plot([dcdt_down, 0, dcdt_up],np.repeat(ys[i], 3), label='%s' % soil.title())

my_xticks = np.log10([0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0])
my_xtick_labels = ["%.2f" % x_tick for x_tick in 10.0**my_xticks]
#g.ax_joint.set_xlim((-28,18))
ax.set_xlim((my_xticks[0],my_xticks[-1]))
ax.set_ylim([0,3.75])
ax.set_xticks(my_xticks)
ax.set_xticklabels(my_xtick_labels)
ax.legend(loc='best')
plt.show()
