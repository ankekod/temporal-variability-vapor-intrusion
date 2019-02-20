import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp2d


# loads data
asu = pd.read_csv('./data/asu_house.csv')
#simulation = pd.read_csv('./data/asu_house.csv')



# 2d interpolation function
#interp_func = interp2d(p_in, Ae, alpha, kind='linear')

fig, ax = plt.subplots()
asu['AirExchangeRate'].plot(kind='kde',ax=ax)


samp = asu['AirExchangeRate'].sample(10)
print(samp.values)


ax.plot(samp.values, np.repeat(0.1, len(samp)), 'o')
#samp['AirExchangeRate'].plot(ax=ax)


plt.show()
