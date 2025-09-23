import numpy as np
import pandas as pd
from numpy import polyfit

path = 'CpVersusTSR&Pitch_WT_Perf_V1.csv'
df = pd.read_csv(path,header=None)
x = df.iloc[:,0].values
y = [df.iloc[:,i+6].values for i in range(6)]
ymax = [np.max(df.iloc[:,i+6].values) for i in range(6)]
x_max = [np.where(df.iloc[:,i+6].values==np.max(df.iloc[:,i+6].values)) for i in range(6)]

coeffs = polyfit(x, y[0], 5)
def Cp_calc(x):
    return coeffs[0] * x**5 + coeffs[1] * x**4 + coeffs[2] * x**3 + coeffs[3] * x**2 + coeffs[4] * x**1 + coeffs[5] * x**0 


class TorqueController:
    def __init__(self):
        self.rho = 1.225 #Air density
        self.Ar = np.pi * 57.5**2 #Rotor area
        self.Rr = 57.5
        self.K = 0
        self.tau_g=0
        self.PG = 0

    def update(self,lmbd_star,wg):
        Cp_star = Cp_calc(lmbd_star)
        K = 0.5*self.rho*self.Ar*(self.Rr**3)*Cp_star/(lmbd_star**3)
        tau_g = K*(wg**2)
        self.K = K
        self.tau_g = tau_g
        self.PG = tau_g*wg