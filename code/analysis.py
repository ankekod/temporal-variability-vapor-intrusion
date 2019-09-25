import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint, ode, solve_ivp
from scipy.optimize import curve_fit, leastsq
from scipy.interpolate import interp1d


class Data:
    def __init__(self, file):
        self.path = file
        return

    def get_path(self):
        return self.path

    def get_data(self):
        return self.data


class COMSOL(Data):
    """
    Class to load, process, and return the COMSOL simulation data
    """

    def __init__(self, file):
        Data.__init__(self, file)
        self.process_raw_data()
        self.A_ck = 0.1 * 4
        return

    def get_crack_area(self):
        return self.A_ck

    def get_raw_data(self):
        path = self.get_path()
        return pd.read_csv(path, header=4)

    def get_renaming_scheme(self):
        renaming = {'% Time (h)': 'time', 'p_in (Pa)': 'p_in', 'alpha (1)': 'alpha',
                    'c_in (ug/m^3)': 'c_in', 'j_ck (ug/(m^2*s))': 'j_ck', 'm_ads (g)': 'm_ads',
                    'c_ads (mol/kg)': 'c_ads',
                    'c_ads/c_gas (1)': 'c_ads/c_gas', 'alpha_ck (1)': 'alpha_ck',
                    'n_ck (ug/s)': 'n_ck', 'Pe (1)': 'Pe', 'c_gas (ug/m^3)': 'c_gas',
                    'u_ck (cm/h)': 'u_ck', '% K_ads (m^3/kg)': 'K_ads',
                    't (h)': 'time', 'c_ads_vol (ug/m^3)': 'c_ads_vol', 'c_liq (ug/m^3)': 'c_liq',
                    '% Pressurization cycles index': 'p_cycle',
                    'Pressurization cycles index': 'p_cycle'}
        return renaming

    def process_raw_data(self):
        raw_df = self.get_raw_data()
        self.data = raw_df.rename(columns=self.get_renaming_scheme())
        return

    def get_time_data(self):
        df = self.get_data()
        return df['time'].values

    def get_entry_flux_data(self):
        df = self.get_data()
        return df['j_ck'].values

    def get_entry_rate_data(self):
        df = self.get_data()
        return df['n_ck'].values

    def get_concentration_data(self):
        df = self.get_data()
        return df['c_in'].values
