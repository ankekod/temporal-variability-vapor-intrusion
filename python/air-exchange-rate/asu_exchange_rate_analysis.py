import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
import seaborn as sns

# returns the error between the predicted and measured TCE in indoor air
def get_error(data):
    error = np.array([])
    n = len(data['day'].values)
    for i in range(n):
        try:
            error = np.append(error,(data['estimate'].values[i]-data['ug/m3'].values[i]))
        except:
            error = np.append(error,np.nan)
    data = data.assign(error=error)
    return data

# generates a data frame that contains the mean and median absolute error
# between the predicted and measured TCE indoor air concentrations
def get_error_table(data):
    max_error = data['error'].abs().max()
    min_error = data['error'].abs().min()
    var = data['error'].var()
    # mean absolute error
    mae = data['error'].abs().mean()
    # medan absolute error
    mdae = data['error'].abs().median()
    # creates a new dataframe containing the error estimates
    error_table = pd.DataFrame(data={'Mean absolute difference': [mae],
                                'Median absolute difference': [mdae],
                                'Max. absolute difference': [max_error],
                                'Min. absolute difference': [min_error],
                                'Variance': [var]})

    return error_table

# estimates the TCE in indoor air by assuming that entry rate is the same as
# measured/calculated but keeping air exchange rate*volume to its median value
def predict_tce_indoor_air(data):
    cpm_start = 780.0
    Q_median_no_cpm = data['building_flow_rate'].loc[data['day'] < cpm_start].median()
    # estimates the TCE in indoor iar
    #tce_indoor_air_prediction = data['tce_emission_rate']/data['building_flow_rate'].median()
    tce_indoor_air_prediction = data['tce_emission_rate']/Q_median_no_cpm
    # assigns predicted value to data dataframe
    data = data.assign(tce_indoor_air_prediction=tce_indoor_air_prediction)

    return data

# method that plots the predicted and measured TCE in indoor air along with the
# measured and median air exchange rate
def plot_comparison(data,title=None,save_figure=False):
    fig, ax1 = plt.subplots(figsize=(10,5))
    # median air exchange rate
    cpm_start = 780.0
    Ae_median = data['exchange_rate'].loc[data['day'] < cpm_start].median()
    # plots air exchange rate
    data.plot(x='day',y='exchange_rate',style='-',color='green',ax=ax1,legend=False)
    # plots median exchange rate as a dahsed line
    t_min, t_max = 125, 1305
    ax1.plot([t_min, t_max], [Ae_median, Ae_median], color='green',linestyle='--')
    # plots the predicted and measured TCE in indoor air
    ax2 = ax1.twinx()
    data.plot(x='day',y=['tce_indoor_air_estimation','tce_indoor_air_prediction'],style='-',mfc='none',logy=True,ax=ax2)
    # labels & legends
    ax1.set_xlabel('Day')
    ax2.set_ylabel('TCE in indoor air $\\mathrm{(\\mu g/m^3)}$')
    ax1.set_ylabel('Air exchange rate (1/day)')

    pipe_closed = 1071.0
    cpm_start, cpm_end = 780.0, 1157.0
    arrow = dict(facecolor='black', shrink=0.05, width=1, headwidth=10)
    ax2.annotate('CPM starts', xy=(cpm_start,1), xytext=(575, 6),arrowprops=arrow)
    ax2.annotate('Land drain\nclosed', xy=(pipe_closed,6), xytext=(1200, 12),arrowprops=arrow)
    ax2.annotate('CPM ends', xy=(cpm_end,0.1), xytext=(1200, 1),arrowprops=arrow)

    handles, labels = ax2.get_legend_handles_labels()
    display = (0,1)
    Ae_legend = plt.Line2D((0,1),(0,0),color='green',linestyle='-')
    Ae_const_legend = plt.Line2D((0,1),(0,0),color='green',linestyle='--')
    ax2.legend(
        labels=(
        'Measured TCE in indoor air',
        'Prediction assuming constant exchange rate',
        'Air exchange rate','Median value = %1.1f (1/day)' % Ae_median
        ),
        handles=handles+[Ae_legend,Ae_const_legend],
        loc='best')
    # if title is given, uses that as the title
    if isinstance(title, str):
        ax1.set_title(title)
    plt.tight_layout()
    # if a figure name is given, the figure is saved
    if isinstance(save_figure, str):
        fig.savefig('./figures/air-exchange-rate/%s.pdf' % save_figure, dpi=300)
        return
    # otherwise it will just show the plot
    else:
        plt.show()
        return

# connects to the database
db = sqlite3.connect('/home/jonathan/lib/vapor-intrusion-dbs/asu_house.db')
data = pd.read_sql_query("SELECT * FROM parameters;", db)
data = data.interpolate(limit_area='inside',limit=250) # 250 limit for pretty "real" dataset, 1500 for fuller plot

# units conversion stuff
# estimated concentration based on emission rates
data['tce_indoor_air_estimation'] = data['tce_emission_rate']/data['building_flow_rate']


data = predict_tce_indoor_air(data)
data = get_error(data)
error_table = get_error_table(data)

M = 131.4
g_to_ug = 1e6
s_to_day = 3600.0*24.0
data['tce_indoor_air_estimation'] *= M*g_to_ug # ug/m3
data['tce_indoor_air_prediction'] *= M*g_to_ug # ug/m3
data['exchange_rate'] *= s_to_day # 1/day



plot_comparison(data,
    title='Predicted TCE in indoor air, assuming constant air exchange rate vs. measurements',
    save_figure='tce_indoor_air_exchange_rate_impact')

# cpm start and end days
cpm_start, cpm_end = 780.0, 1157.0

data_filter_no_cpm = data[(data['day'] < cpm_start) | (data['day'] > cpm_end)]
data_filter_only_cpm = data[(data['day'] > cpm_start) & (data['day'] < cpm_end)]


# all of the dataset
len_data = len(data['tce_indoor_air_estimation'])
len_predict = len(data['tce_indoor_air_prediction'])
new_df = pd.DataFrame({'TCE in indoor air': data['tce_indoor_air_estimation'].values,
            'Period': np.repeat('Full',len_data),
            'Exchange rate': np.repeat('Exchange rate fluctating', len_data)})
new_df2 = pd.DataFrame({'TCE in indoor air': data['tce_indoor_air_prediction'].values,
            'Period': np.repeat('Full',len_data),
            'Exchange rate': np.repeat('Exchange rate constant', len_data)})

# no cpm dataset
len_data = len(data_filter_no_cpm['tce_indoor_air_estimation'])
len_predict = len(data_filter_no_cpm['tce_indoor_air_prediction'])
new_df3 = pd.DataFrame({'TCE in indoor air': data_filter_no_cpm['tce_indoor_air_estimation'].values,
            'Period': np.repeat('CPM excluded',len_data),
            'Exchange rate': np.repeat('Exchange rate fluctating', len_data)})


new_df4 = pd.DataFrame({'TCE in indoor air': data_filter_no_cpm['tce_indoor_air_prediction'].values,
            'Period': np.repeat('CPM excluded',len_predict),
            'Exchange rate': np.repeat('Exchange rate constant', len_predict)})

# cpm only dataset
len_data = len(data_filter_only_cpm['tce_indoor_air_estimation'])
len_predict = len(data_filter_only_cpm['tce_indoor_air_prediction'])
new_df5 = pd.DataFrame({'TCE in indoor air': data_filter_only_cpm['tce_indoor_air_estimation'].values,
            'Period': np.repeat('CPM only',len_data),
            'Exchange rate': np.repeat('Exchange rate fluctating', len_data)})


new_df6 = pd.DataFrame({'TCE in indoor air': data_filter_only_cpm['tce_indoor_air_prediction'].values,
            'Period': np.repeat('CPM only',len_predict),
            'Exchange rate': np.repeat('Exchange rate constant', len_predict)})



df = pd.concat([new_df, new_df2, new_df3, new_df4, new_df5, new_df6],axis=0)
df = df.dropna()


df['TCE in indoor air'] = np.log10(df['TCE in indoor air'].values)


fig, ax = plt.subplots()
sns.violinplot(x='Period', y='TCE in indoor air', hue='Exchange rate', data=df, split=True, inner="quartile", ax=ax)
ax.set_ylabel('TCE in indoor air $\\mathrm{(\\mu g/m^3)}$')
ax.legend(loc='lower right')

ax.set_yticks([-3, -2, -1, 0, 1, 2])
ax.set_yticklabels(['0.001', '0.01', '0.1', '1.0', '10.0', '100.0'])

ax.set_title('Fluctating vs. constant air exchange rate\'s impact\non distribution of TCE in indoor air')

plt.savefig('./figures/air-exchange-rate/violin-asu-iacc.pdf', dpi=300)
fig.clf()
