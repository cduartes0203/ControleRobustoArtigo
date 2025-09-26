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
        self.PG_ = np.array([0])
        self.PD_ = np.array([0])
        self.ED_ = np.array([0])
        self.EG_ = np.array([0])
        self.tR = 1e-10
        self.tG = 1e-10
        self.Cp_star = params['Cp_Max']
        self.lambda_star = params['Lambda_opt']
        self.K_mppt = 0
        self.X = np.array([1e-10,1e-10,1e-10]).reshape(-1,1)
        self.dX =  np.array([1e-10,1e-10,1e-10]).reshape(-1,1)
        self.k = 0

    def update(self, v, dt, uk=0, **params):

        rho = params['rho'] 
        Ar = params['Ar'] 
        Rr = params['Rr'] 
        B_dt = params['Bdt']
        K_dt = params['Kdt']
        J_r = params['Jr']
        J_g = params['Jg']
        wr = self.X[0]
        wg = self.X[1]

        lmbd = wr*Rr/v
        Cp = Cp_calc(lmbd)

        self.lambda_star = self.lambda_star + uk
        self.Cp_star = Cp_calc(self.lambda_star)
        self.K_mppt = 0.5*rho*Ar*(Rr**3)*self.Cp_star/(self.lambda_star**3)

        if self.k >0:
            self.tR = 0.5*rho*Ar*Cp*(v**3)/wr
            self.tG = self.K_mppt*(wg**2)
      

        A=np.array([[-B_dt/J_r,  B_dt/J_r, -K_dt/J_r],
                    [ B_dt/J_g, -B_dt/J_g,  K_dt/J_g],
                    [        1,        -1,         0]])
        
        B = np.array([[1/J_r,      0],
                      [    0, -1/J_g],
                      [    0,      0]])
        
        tau = np.array([self.tR, self.tG]).reshape(-1,1) 
        self.dX = (A@self.X)+(B@tau)
        self.X = self.X + self.dX * dt
        
        self.PD = B_dt*((wr - wg)**2)
        self.PG = self.tG*wg
        self.ED = self.ED + self.PD
        self.EG = self.EG + self.PG

        self.EG_ = np.append(self.EG_, self.EG)
        self.ED_ = np.append(self.ED_, self.ED)
        self.PG_ = np.append(self.PG_, self.PG)
        self.PD_ = np.append(self.PD_, self.PD)

        self.k = self.k + 1

        return self.ED

    def show_attributes(self):
        print('Torque Controller parameters:')
        df = pd.DataFrame()
        for key, value in self.__dict__.items():
            df[key] = [value]
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', 100) 
        print(df)