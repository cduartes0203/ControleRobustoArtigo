import numpy as np
import pandas as pd
from numpy import polyfit

def Cp_calc(lmbd_in=0):
    path = 'Cp_X_lambda.csv'
    df = pd.read_csv(path)
    dist = np.abs(df['lambda'].values - lmbd_in)

    return (df['cp'].values[np.argmin(dist)])

class TorqueController:
    def __init__(self):
        self.rho = 1.225 #Air density
        self.Ar = np.pi * 57.5**2 #Rotor area
        self.Rr = 57.5
        self.K = 0
        self.tau_g=0
        self.PG = 0
        self.Cp_star = 0

    def update(self,lmbd_star,wg, prnt=False):
        Cp_star = Cp_calc(lmbd_star)
        K = 0.5*self.rho*self.Ar*(self.Rr**3)*Cp_star/(lmbd_star**3)
        tau_g = K*(wg**2)
        self.K = K
        self.tau_g = tau_g
        self.PG = tau_g*wg
        self.Cp_star = Cp_star

        if prnt: self.show_attributes()

    def show_attributes(self):
        print('________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________')
        print('Torque Controller parameters:')
        df = pd.DataFrame()
        for key, value in self.__dict__.items():
            df[key] = [value]
            #print(f"{key}: {value}")
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', 100) 
        print(df)