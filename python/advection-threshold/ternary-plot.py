import numpy as np
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.io as pio

import json
from urllib import request

url = 'https://gist.githubusercontent.com/davenquinn/988167471993bc2ece29/raw/f38d9cb3dd86e315e237fde5d65e185c39c931c2/data.json'
response = request.urlopen(url).read()
data = json.loads(response)

colors = ['#8dd3c7','#ffffb3','#bebada',
          '#fb8072','#80b1d3','#fdb462',
          '#b3de69','#fccde5','#d9d9d9',
          '#bc80bd','#ccebc5','#ffed6f'];


pressure = {
    'sand': '8.0',
    'loamy sand': '45',
    'sandy loam': '34',
    'sandy clay loam': '234',
    'sandy clay': '223',
    'clay': '44',
    'clay loam': '34',
    'silty clay': '43',
    'silty clay loam': '12',
    'silty loam': '13',
    'silt': '87',
    'loam': '78',
    }


# generate a,b and c from JSON data..
traces = []
color_iter = iter(colors)
for i in data.keys():
    trace = dict(text=i,
        type='scatterternary',
        a=[ k['clay'] for k in data[i] ],
        b=[ k['sand'] for k in data[i] ],
        c=[ k['silt'] for k in data[i] ],
        mode='lines',
        line=dict(color='#444'),
        fill='toself',
        fillcolor=next(color_iter),
        name = pressure[i],
        showlegend = True,
    )
    traces.append(trace)

layout = {
    'title': 'Soil types',
    'ternary':
        {'sum':100,
         'aaxis':{'title': 'clay', 'ticksuffix':'%', 'min': 0.01, 'linewidth':2, 'ticks':'outside' },
         'baxis':{'title': 'sand', 'ticksuffix':'%', 'min': 0.01, 'linewidth':2, 'ticks':'outside' },
         'caxis':{'title': 'silt','ticksuffix':'%', 'min': 0.01, 'linewidth':2, 'ticks':'outside' }},
    'showlegend': False,
    'annotations': [{
      'showarrow': False,
      'text': 'Clay',
        'x': 0.5,
        'y': 0.75,
        'font': { 'size': 15 }
    }]
}

fig = dict(data=traces, layout=layout)


#plot(fig)
pio.write_image(fig, 'figures/cpm-ternary.pdf')
