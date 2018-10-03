import plotly.graph_objs as go
import plotly.offline as py
import plotly.io as pio
import pandas as pd
import matplotlib.pyplot as plt
data_dir = './data/advection-threshold/'
figures_dir = './figures/advection-threshold/'

def radar_plot(dfs, filename):
    # soil list
    soils = (
        'Sand',
        'Loamy Sand',
        'Sandy Loam',
        'Sandy Clay Loam',
        'Loam',
        'Silt Loam',
        'Clay Loam',
        'Silty Clay Loam',
        'Silty Clay',
        'Silt',
        'Sandy Clay',
        'Clay',
    )

    data = []
    for i in range(len(df.columns)):
        print(i)
        data.append(
            go.Scatterpolar(
                r = df.values[:,i]*-1,
                theta = soils,
                fill = 'toself',
                name = "%s = %1.1f" % (df.columns.name, df.columns.values[i])
            )
        )

    layout = go.Layout(
      polar = dict(
        radialaxis = dict(
          visible = True,
        )
      ),
      showlegend = True
    )

    fig = go.Figure(data=data, layout=layout)
    pio.write_image(
        fig,
        file = figures_dir + filename + '.pdf',
        )
    return

df = pd.read_csv(data_dir+'perimeter-ck-gravel-sub-base.csv', header=4)

print(df.pivot(index='Soil type', columns='L', values='Indoor/outdoor pressure difference').index.values)

radar_plot(
    filename='test',
    dfs = [df.pivot(index='Soil type', columns='L', values='Indoor/outdoor pressure difference')]
)
