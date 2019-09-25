import numpy as np
import pandas as pd


def sub_slab_attenuation():
    df = get_asu_house_data().set_index('CPM')

    sns.boxplot(x="CPM", y="alpha_slab", data=df)


    fig, ax = plt.subplots(dpi=300)
    return
