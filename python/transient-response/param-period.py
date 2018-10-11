import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Ae = 12.0 # 1/day
M = 131.38 # TCE g/mol
V = 300.0 # basement volume m3

def Ae(p):
    return 12.0*p

def dp(t, period):
    dp = 10.0*np.sin(2.0*np.pi*t/period) - 0.5
    return dp

df = pd.read_csv('./data/transient-response/param-period2.csv', header=4)
df['Pressure'] = dp(df['Time'], df['period'])
df['TCE expulsion rate'] = df['TCE in indoor air']*Ae(df['Pressure'])*M*V


df2 = pd.read_csv('data/preferential-pathway-sensitivity/param-preferential-pathway.csv', header=4)
reference_lines = df2[((df2['chi'] == 1) & (df2['SB']==1)) & ((df2['p'] == 10.5) | (df2['p'] == -10.5))]

print(reference_lines)


pivoted_df = df.pivot(
    index='Time',
    columns='period',
    values=['Attenuation factor','Pressure','TCE emission rate','TCE expulsion rate'],
).interpolate(
    method='piecewise_polynomial',
)

for period in df['period'].unique():
    print(df['Pressure'][df['period']==period].median())

fig, ax1 = plt.subplots()

#ax2 = ax1.twinx()

pivoted_df.plot(
    logy=True,
    logx=True,
    y=['TCE expulsion rate'],
    #secondary_y='Pressure',
    ax=ax1,
)

t0 = 0
tau = pivoted_df.index.values[-1]


ax1.loglog(
    [t0, tau],
    [df2['alpha'].min(), df2['alpha'].min()],
    'k--',
)

ax1.loglog(
    [t0, tau],
    [df2['alpha'].max(), df2['alpha'].max()],
    'k--',
)

# check what the average p is over each period, I think it should be higher for
# the larger sin waves? could be interesting to see if you essentially achieve
# the ss alpha value that corresponds to this average pressure value ?


plt.show()
