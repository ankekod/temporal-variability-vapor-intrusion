import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import LSODA
from scipy.interpolate import interp1d
import sqlite3
import seaborn as sns

db_dir = '/home/jonathan/lib/vapor-intrusion-dbs/'
#db_dir = 'C://Users/jstroem/lib/vapor-intrusion-dbs/'

db = sqlite3.connect(db_dir + 'hill-afb.db')

asu = pd.read_sql_query("SELECT * from daily_averages;", db,
                        ).sort_values('time')
asu.time = asu.time.apply(pd.to_datetime)

asu['pressure'] *= -1
asu['dc'] = asu['concentration'].apply(np.log10).diff()
asu['dp'] = asu['pressure'].diff()
asu = asu.dropna()

asu['csign'] = asu['dc'].apply(np.sign)
asu['psign'] = asu['dp'].apply(np.sign)

asu['csigncumsum'] = asu['csign'].cumsum()
asu['psigncumsum'] = asu['psign'].cumsum()
asu['ccumsum'] = asu['dc'].cumsum()
asu['pcumsum'] = asu['dp'].cumsum()


#asu['concentration'] = asu['concentration'].apply(np.log10)
#asu = asu.replace([np.inf, -np.inf], np.nan).dropna()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

asu.plot(x='time', y='concentration', color='red', logy=True, ax=ax1)
asu.plot(x='time', y=['csigncumsum', 'psigncumsum'], ax=ax2)


sns.jointplot(asu['psigncumsum'], asu['concentration'], kind='reg')

plt.show()
