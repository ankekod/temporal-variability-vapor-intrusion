import plotly.graph_objs as go
import plotly.offline as py
import plotly.io as pio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_dir = './data/advection-threshold/'
figures_dir = './figures/advection-threshold/'

def radar_plot(df, filename):
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


for file in ('perimeter-ck-gravel-sub-base', 'perimeter-ck-uniform-soil'):
    df = pd.read_csv(data_dir+file+'.csv', header=4)
    for col in ('GW depth','Base depth','Crack width'):
        filename = '%s-%s' % (file, col.lower().replace(' ','-'))
        print('Generating %s plot' % filename)
        radar_plot(
            filename=filename,
            df = df.pivot_table(index='Soil type', columns=col, values='Indoor/outdoor pressure difference', aggfunc=np.median),
        )
