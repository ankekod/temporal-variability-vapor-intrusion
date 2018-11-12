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
        td.*, \
        gw.concentration as gw_concentration \
    FROM \"td-basement\" td\
    LEFT JOIN groundwater_concentration gw ON (date(td.time) == date(gw.time))\
    UNION ALL\
    SELECT\
        td.*, \
        gw.concentration as gw_concentration \
    FROM groundwater_concentration gw\
    LEFT JOIN \"td-basement\" td ON (date(td.time) == date(gw.time))\
    WHERE td.time IS NULL;",
    db,
).replace(
    0,
    np.nan,
).interpolate(
    method='piecewise_polynomial',
).dropna()


#asu = asu.dropna()
asu.time = asu.time.apply(pd.to_datetime)

K_H = 0.403
asu['alpha'] = asu['concentration']/(asu['gw_concentration']*1e3*K_H)


phases = pd.read_sql_query("SELECT * from phases;", db)


asu['month'] = asu['time'].map(lambda x: x.month) # assign integer for each month

seasons = {
    'winter': (12,2),
    'spring': (3,5),
    'summer': (6,8),
    'fall': (9,11),
}

def get_season(x):
    if (x==12) or (x==1) or (x==2):
        return 'winter'
    elif (x==3) or (x==4) or (x==5):
        return 'spring'
    elif (x==6) or (x==7) or (x==8):
        return 'summer'
    elif (x==9) or (x==10) or (x==11):
        return 'fall'
    else:
        return 'error'
asu['season'] = asu['month'].apply(lambda x: get_season(x))
print(asu['season'].unique())

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

pre_cpm = pre_cpm.replace([np.inf, -np.inf], np.nan).dropna()

post_cpm = post_cpm.replace([np.inf, -np.inf], np.nan).dropna()



df = pd.read_csv('./data/transient-response/cstr-changes.csv')
mol_m3_to_ug_m3 = 131.4*1e6
df[['c_max','c_min']] *= mol_m3_to_ug_m3
df[['c_max','c_min']] = df[['c_max','c_min']].apply(np.log10)
df[['t_down','t_up']] /= 24.0
df['dcdt_up'] = (df['c_max']-df['c_min'])/df['t_up']
df['dcdt_down'] = (df['c_min']-df['c_max'])/df['t_down']

df = df[df['Ae']==0.5]
print(len(pre_cpm),len(post_cpm))
def custom_x_ticks(ax):
    my_xticks = np.log10([0.01,0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0,100.0])
    my_xtick_labels = ["%.2f" % x_tick for x_tick in 10.0**my_xticks]
    #ax.set_xlim((my_xticks[0],my_xticks[-1]))
    ax.set_xticks(my_xticks)
    ax.set_xticklabels(my_xtick_labels)
    ax.set_xlabel('$\\frac{\\Delta \\log(\\alpha)}{\\Delta t}$')
    ax.set_ylim([0,1.5])
    return

fig, ax = plt.subplots()
print(pre_cpm['dalphadt'])
sns.distplot(pre_cpm['dalphadt'],ax=ax,label='c, PP open')
sns.kdeplot(post_cpm['dalphadt'],ax=ax,label='c, PP closed')


ys = np.linspace(0.5,3.0,len(df.index))
df2 = df[df['soil']=='sandy-clay'].copy()
for i, soil, dcdt_down, dcdt_up in zip(df2.index, df2.soil, df2.dcdt_down, df2.dcdt_up):
    ax.plot([dcdt_down, 0, dcdt_up],np.repeat(ys[i], 3), label='%s' % soil.title())

custom_x_ticks(ax)
ax.legend(loc='best')
#plt.show()


fig, ax = plt.subplots()
for season in asu['season'].unique():
    sns.kdeplot(pre_cpm[pre_cpm['season']==season]['dalphadt'],ax=ax,label='%s' % season.title())
    #sns.kdeplot(post_cpm[post_cpm['season']==season]['dalphadt'],ax=ax,label='c, PP closed')
custom_x_ticks(ax)
ax.set_title('PP Open')

fig, ax = plt.subplots()
for season in asu['season'].unique():
    if season != 'summer':
        sns.kdeplot(post_cpm[post_cpm['season']==season]['dalphadt'],ax=ax,label='%s' % season.title())
    else:
        continue
    #sns.kdeplot(post_cpm[post_cpm['season']==season]['dalphadt'],ax=ax,label='c, PP closed')
custom_x_ticks(ax)
ax.set_title('PP Closed')

plt.show()
