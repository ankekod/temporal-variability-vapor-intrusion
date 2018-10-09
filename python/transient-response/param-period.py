import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def dp(t, period):
    dp = 10.0*np.sin(2.0*np.pi*t/period) - 0.5
    return dp

df = pd.read_csv('./data/transient-response/param-period2.csv', header=4)



df[['Time', 'period']].apply(lambda x: print(x[0][0]))
#df['Pressure'] = df['']


"""
pivoted_df = df.pivot(
    index='Time',
    columns='period',
    values='Attenuation factor',
)

pivoted_df.interpolate(
    method='piecewise_polynomial',
).plot(
    logy=True,
    logx=True,
)
# check what the average p is over each period, I think it should be higher for
# the larger sin waves? could be interesting to see if you essentially achieve
# the ss alpha value that corresponds to this average pressure value ?


#plt.show()
"""
