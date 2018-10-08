import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('./data/transient-response/param-period.csv', header=4)

print(df.pivot( columns='period', values='Attenuation factor'))
