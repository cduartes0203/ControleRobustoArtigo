import numpy as np
import pandas as pd


class ReferenceConverter:
    def __init__(self, RUL_ref=1e6):
        self.rul_ref = RUL_ref
        self.D_max = 1000
        self.beta_ref = 1e-3
    
    def update(self, D_hat, k, prnt=False):
        self.rul_ref = self.rul_ref-k
        self.beta_ref = (self.D_max-D_hat)/self.rul_ref
        if prnt: self.show_attributes()
        

    def show_attributes(self):
        print('________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________')
        print('Reference Converter parameters:')
        df = pd.DataFrame()
        for key, value in self.__dict__.items():
            df[key] = [value]
            #print(f"{key}: {value}")
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', 100) 
        print(df)