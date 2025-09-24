import numpy as np
import pandas as pd
import cvxpy as cp # type: ignore

class StateObserver:
    def __init__(self,Di=1, Bi=1e-3, ts=1):
        """
        Inicializa o Observador de Estado.
        ts: Período de amostragem (Ts ou dt).
        """
        self.ts = ts
        # Vetor de estado estimado inicial: [D_hat, beta_hat]
        self.x_hat = np.array([Di, Bi])
        
        # Ganho do Observador, L. Será calculado pelo método solve_lmi_gain.
        self.L = None
        
        # Matrizes do observador para o estudo de caso (Equação 71)
        # H = [[1, ts], [0, 1]]
        self.H = np.array([[1, ts], [0, 1]])
        # C = [1, 0]
        self.C = np.array([1, 0]).reshape(1, 2)
        
    def solve_lmi_gain(self, Q=np.diag([1, 2.5e-5]), R=np.array([[100.0]])):
        """
        Calcula o ganho ótimo do observador L usando a solução LMI para o problema LQ
        proposto na Seção 4.4.3 do artigo.

        Args:
            Q (np.array): Matriz de custo do estado (process noise covariance).
            R (np.array): Matriz de custo da medição (measurement noise covariance).
        """
        n = self.H.shape[0] # Dimensão do estado (n=2)
        p = self.C.shape[0] # Dimensão da saída (p=1)

        # ---- Formulação do Problema LMI (Seção 4.4.3) ----
        # 1. Definir as variáveis da LMI
        # P: Matriz de Lyapunov (simétrica, definida positiva)
        P = cp.Variable((n, n), symmetric=True)
        # Y: Variável de mudança para L (L = inv(P) * Y^T)
        Y = cp.Variable((p, n))
        # W: Limite superior para a função de custo
        W = cp.Variable((n + p, n + p), symmetric=True)

        # 2. Definir as matrizes de custo estendidas M e N (Equação 56)
        M = np.vstack([np.sqrt(Q), np.zeros((p, n))])
        N = np.vstack([np.zeros((n, p)), np.sqrt(R)])

        # 3. Definir as restrições (constraints) da LMI
        constraints = [P >> 1e-16*np.eye(n)] # P deve ser definida positiva

        # Restrição de estabilidade (Equação 58, adaptada da condição de Lyapunov)
        # Esta é a LMI principal que garante a convergência do erro de estimação.
        lmi_stability = cp.bmat([
            [(-P+Q), (self.H.T @ P - self.C.T @ Y)],
            [(P @ self.H - Y.T @ self.C), P]
        ])
        constraints += [lmi_stability >> 0]

        # Restrição da função de custo (Equação 61)
        lmi_cost = cp.bmat([
            [W, (M @ P + N @ Y)],
            [(P @ M.T + Y.T @ N.T), P]
        ])
        constraints += [lmi_cost >> 0]
        
        # 4. Definir o problema de otimização (Equação 62)
        # Minimizar o traço de W, que minimiza a função de custo do erro.
        objective = cp.Minimize(cp.trace(W))
        problem = cp.Problem(objective, constraints)

        # 5. Resolver o problema
        problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            # Calcular o ganho L a partir das variáveis da solução
            # L = (inv(P) * Y^T)
            P_val = P.value
            Y_val = Y.value
            L_val = np.linalg.inv(P_val) @ Y_val.T
            self.L = L_val.flatten()
            #print(f"Ganho L calculado com sucesso: {self.L}")
            return self.L
        else:
            print(f"Falha ao resolver a LMI. Status: {problem.status}")
            return None

    def update_state(self, y_k, prnt=False):
        """
        Atualiza o estado estimado usando a equação do observador (Equação 70).

        Args:
            y_k (float): A medição atual da degradação (energia dissipada).
        
        Returns:
            np.array: O novo vetor de estado estimado x_hat_{k+1}.
        """
        if self.L is None:
            raise RuntimeError("O ganho L não foi calculado. Chame 'solve_lmi_gain' primeiro.")
        
        # Equação 70: x_hat_{k+1} = H * x_hat_k + L * (y_k - C * x_hat_k)
        x_hat_k = self.x_hat.reshape(-1, 1)
        L_reshaped = self.L.reshape(-1, 1)
        
        # Termo de correção
        estimation_error = y_k - self.C @ x_hat_k
        correction_term = L_reshaped * estimation_error
        
        # Próximo estado estimado
        x_hat_k_plus_1 = self.H @ x_hat_k + correction_term
        
        self.x_hat = x_hat_k_plus_1.flatten()
        
        if prnt: self.show_attributes()
        
        return self.x_hat
    
    def show_attributes(self):
        print('________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________')
        print('State Observer parameters:')
        df = pd.DataFrame()
        for key, value in self.__dict__.items():
            df[key] = [value]
            #print(f"{key}: {value}")
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', 100) 
        print(df)
