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
        self.tau_r = 0
        self.tau_g = 0
        self.Cp_star = params['Cp_Max']
        self.lambda_star = params['Lambda_opt']
        self.K_mppt = None
        self.sttsB = np.array([1e-10,1e-10,1e-10]).reshape(-1,1)
        self.sttsA = np.array([1e-10,1e-10,1e-10]).reshape(-1,1)
        self.dstts =  np.array([1e-10,1e-10,1e-10]).reshape(-1,1)

    def update(self, v, dt, uk=0, **params):

        rho = params['rho'] 
        Ar = params['Ar'] 
        Rr = params['Rr'] 
        B_dt = params['Bdt']
        K_dt = params['Kdt']
        J_r = params['Jr']
        J_g = params['Jg']
        Cp_Max = params['Cp_Max']
        Lambda_opt = params['Lambda_opt']
        wr = self.sttsB[0]
        wg = self.sttsB[1]
        ot = self.sttsB[2]

        self.sttsB = self.sttsA

        lmbd = wr*Rr/v
        Cp = Cp_calc(lmbd)
        tau_r = 0.5*rho*Ar*Cp*(v**3)/wr
        if wr < 0: tau_r = 0
        lambda_star = Lambda_opt + uk

        
        Cp_star = Cp_calc(lambda_star)
        self.Cp_star = Cp_star
        self.lambda_star = lambda_star

        self.K_mppt = 0.5*rho*Ar*(Rr**3)*Cp_star/(lambda_star**3)
        
        
        tau_g = self.K_mppt*(wg**2)
        self.tau_g = tau_g
        self.tau_r = tau_r

        A=np.array([[-B_dt/J_r,  B_dt/J_r, -K_dt/J_r],
                    [ B_dt/J_g, -B_dt/J_g,  K_dt/J_g],
                    [        1,        -1,         0]])
        
        B = np.array([[1/J_r,      0],
                      [    0, -1/J_g],
                      [    0,      0]])
        
        tau = np.array([tau_r, tau_g]).reshape(-1,1) 
        d_states = (A@self.sttsB)+(B@tau)
        #print('d_states:',d_states, d_states*dt)
        self.sttsA = self.sttsB + d_states * dt
        self.dstts = d_states
        
        self.PD = B_dt*((wr - wg)**2)
        self.PG = tau_g*wg
        self.ED = self.ED + self.PD
        self.EG = self.EG + self.PG

        self.EG_ = np.append(self.EG_, self.EG)
        self.ED_ = np.append(self.ED_, self.ED)
        self.PG_ = np.append(self.PG_, self.PG)
        self.PD_ = np.append(self.PD_, self.PD)

        return self.ED

    def show_attributes(self):
        print('Torque Controller parameters:')
        df = pd.DataFrame()
        for key, value in self.__dict__.items():
            df[key] = [value]
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', 100) 
        print(df)