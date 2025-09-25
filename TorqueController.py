import numpy as np
import pandas as pd
from numpy import polyfit

def Cp_calc(lmbd_in=0):
    path = 'Cp_X_lambda.csv'
    df = pd.read_csv(path)
    dist = np.abs(df['lambda'].values - lmbd_in)

    return (df['cp'].values[np.argmin(dist)])

class TorqueController:
    def __init__(self, **params):

        self.PG = None
        self.PD = None
        self.ED = 0
        self.EG = 0
        self.Cp_star = params['Cp_Max']
        self.Lambda_star = params['Lambda_opt']
        self.K_mppt = None
        self.states = np.array([1,1,1e-8]).reshape(-1,1)

    def update(self, v, dt, Lambda_star=None, **params):

        rho = params['rho'] 
        Ar = params['Ar'] 
        Rr = params['Rr'] 
        B_dt = params['Bdt']
        K_dt = params['Kdt']
        J_r = params['Jr']
        J_g = params['Jg']
        Cp_Max = params['Cp_Max']
        Lambda_opt = params['Lambda_opt']
        wr = self.states[0]
        wg = self.states[1]
        ot = self.states[2]

        lmbd = wr*Rr/v
        Cp = Cp_calc(lmbd)

        if Lambda_star!= None:
            self.Cp_star = Cp_calc(self.Lambda_star)

        K_mppt = 0.5*rho*Ar*(Rr**3)*self.Cp_star/(self.Lambda_star**3)
        self.K_mppt = K_mppt

        tau_r = 0.5*rho*Ar*Cp*(v**3)/wr
        tau_g = K_mppt*(wg**2)

        self.PD = B_dt*((wr - wg)**2)
        self.PG = tau_g*wg
        #print('v:',v,'Cp:',Cp,'\ntau_g:',tau_g,'\nwg:',wg,'\nK_mppt:',K_mppt,'\nPD:',self.PD,'\nPG:',self.PG)
        self.ED = self.ED + self.PD
        self.EG = self.EG + self.PG

        A=np.array([[-B_dt/J_r,  B_dt/J_r, -K_dt/J_r],
                    [ B_dt/J_g, -B_dt/J_g,  K_dt/J_g],
                    [        1,        -1,         0]])
        
        B = np.array([[1/J_r,      0],
                      [    0, -1/J_g],
                      [    0,      0]])
        
        tau = np.array([tau_r, tau_g]).reshape(-1,1) 
        d_states = A@self.states+B@tau
        #print('d_states:',d_states.T)
        self.states = self.states + d_states * dt

        return self.ED

    def show_attributes(self):
        print('Torque Controller parameters:')
        df = pd.DataFrame()
        for key, value in self.__dict__.items():
            df[key] = [value]
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', 100) 
        print(df)