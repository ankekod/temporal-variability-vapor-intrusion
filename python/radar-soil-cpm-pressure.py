import plotly.graph_objs as go
import plotly.offline as py
import plotly.io as pio
import pandas as pd

df = pd.read_csv('./data/cpm-pressure-study/cpm-pressure-soil-type.csv')

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


data = [
    go.Scatterpolar(
      r = df['Indoor/outdoor pressure difference'].values*-1,
      theta = soils,
      fill = 'toself',
      name = 'Indoor/outdoor'
    ),
    go.Scatterpolar(
      r = df['Sub-slab pressure difference'].values*-1,
      theta = soils,
      fill = 'toself',
      name = 'Across slab'
    )
]

layout = go.Layout(
  polar = dict(
    radialaxis = dict(
      visible = True,
      #range = [0, 1954.3]
    )
  ),
  showlegend = True
)

fig = go.Figure(data=data, layout=layout)
pio.write_image(fig, 'figures/radar-soil-cpm-pressure.pdf')
