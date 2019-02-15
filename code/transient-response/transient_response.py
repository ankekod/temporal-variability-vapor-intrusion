import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
from scipy import stats
from scipy.interpolate import interp1d
from scipy.fftpack import fft
from natu import units

# path where the datafiles are stored
file_path = '/Users/duckerfeller/Dropbox/Research/Data/'

data = pd.read_csv(
    file_path+'data_transient.csv', # full path
    header=4, # dataframe headers are in row 4
)

p = lambda t: 10.0*np.sin(2.0*np.pi*t)-0.5

data = data.assign(
    p = p(data.t.values)
)

data.t *= 24.0


max_alpha_loc = data.alpha.idxmax()
min_alpha_loc = data.alpha.idxmin()



# plotting
fig, ax1 = plt.subplots(figsize=(10,5))
ax2 = ax1.twinx()

data.plot(
    x='t',
    y='p',
    ax=ax1,
    color='orange',
    legend=False,
    label='Indoor-outdoor pressure difference (Pa)'
)

data.plot(
    x='t',
    y='alpha',
    logy=True,
    ax=ax2,
    legend=False,
    label='Attenuation factor'
)

ax2.set_title(
    'TCE in indoor air response following change in indoor-outdoor pressurization\n' +
    'Max./min. attenuation factor reached ~0.5 hr after pressure max./min..'
)

# x & y labels
ax1.set_xlabel('Hour')
ax1.set_ylabel('Indoor-outdoor pressure difference (Pa)')
ax2.set_ylabel('Attenuation factor')


# legend stuff
handles = []
labels = []
for ax in (ax1, ax2):
    handle, label = ax.get_legend_handles_labels()
    handles.append(handle)
    labels.append(label)

plt.legend(
    np.array(handles).flatten(),
    np.array(labels).flatten(),
    loc='upper left',
)

print('Delay to max conc: %1.2f (h)' %(data.t.values[max_alpha_loc] - 42.0))
print('Delay to min conc: %1.2f (h)' %(data.t.values[min_alpha_loc] - 30.0))

#plt.show()
plt.tight_layout()
plt.savefig(
    '../Figures/alpha_transient.png',
    dpi=300,
)
