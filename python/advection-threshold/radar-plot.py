import plotly.graph_objs as go
import plotly.offline as py
import plotly.io as pio
import pandas as pd

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
    for df in dfs:
        data.append(
            go.Scatterpolar(
                r = df*-1,
                theta = soils,
                fill = 'toself',
                name = df.name
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
        #format='pdf',
        )
    return

df = pd.read_csv(data_dir+'cpm-pressure-soil-type.csv')
radar_plot(
    filename = 'indoor-outdoor-and-sub-slab-pressure-difference',

    dfs = (
        df['Indoor/outdoor pressure difference'],
        df['Sub-slab pressure difference'],
    ),
)
