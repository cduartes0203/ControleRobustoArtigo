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
        self.B_dt = 755.49 #D_exp_exp_exp_exp_exp_exp_exp_expamping of the drivetrain
        self.K_dt = 2.7*(1e9) #Stiness of the drivetrain
        self.J_r = 55*(1e6) #Inertia of the rotor
        self.J_g = 55*(1e6)
        self.K_mppt = 0
        self.tau_g=0
        self.tau_r=0
        self.PG = 0
        self.PD = 0
        self.ED = 1e-16
        self.EG = 1e-16
        self.Cp_star = 0
        self.wr = 1.5
        self.wg = 1
        self.ot = 1e-16
        self.states = np.array([self.wr, self.wg, self.ot]).reshape(-1,1)




    def update(self,v, lmbd_star, prnt=False):

        lmbd = self.wr*self.Rr/v
        Cp = Cp_calc(lmbd)
        tau_r = 0.5*self.rho*self.Ar*Cp*(v**3)/self.wr

        self.PD = self.B_dt*((self.wr - self.wg)**2)

        self.ED = self.ED + self.PD

        Cp_star = Cp_calc(lmbd_star)
        K_mppt = 0.5*self.rho*self.Ar*(self.Rr**3)*Cp_star/(lmbd_star**3)
        tau_g = K_mppt*(self.wg**2)

        self.PG = tau_g*self.wg

        A=np.array([[-self.B_dt/self.J_r, self.B_dt/self.J_r, -self.K_dt/self.J_r],
                         [self.B_dt/self.J_g, -self.B_dt/self.J_g, self.K_dt/self.J_g],
                         [1, -1, 0]])
        
        B = np.array([[1/self.J_r, 0],
                           [0, -1/self.J_g],
                           [0, 0]])
        
        states = np.array([self.wr, self.wg, self.ot]).reshape(-1,1)
        tau = np.array([tau_r, tau_g]).reshape(-1,1) 
        d_states = A@states+B@tau

        self.wr = self.wr+d_states[0]
        self.wg = self.wg+d_states[1]
        self.ot = self.ot+d_states[2]

        self.states = np.array([self.wr, self.wg, self.ot]).reshape(-1,1)
        
        
        self.K_mppt = K_mppt
        self.tau_r = tau_r
        self.tau_g = tau_g
        self.PG = tau_g*self.wg
        self.Cp_star = Cp_star
        self.EG = self.EG + self.PG

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