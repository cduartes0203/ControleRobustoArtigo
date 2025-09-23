import numpy as np

class ReferenceConverter:
    def __init__(self, RUL_ref=1e6):
        self.rul_ref = RUL_ref
        self.D_max = 1000
        self.beta_ref = 1e-3
    
    def update(self, D_hat, k):
        self.rul_ref = self.rul_ref-k
        self.beta_ref = (self.D_max-D_hat)/self.rul_ref