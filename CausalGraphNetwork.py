import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.linear_model import LinearRegression


class CausalEffectAnalyzer:
    def __init__(self, data: pd.DataFrame, edges: list, variables: list,
                 time_steps: int = 100, step_size: int = 5):

        self.data = data
        self.edges = edges
        self.variables = variables
        self.time_steps = time_steps
        self.step_size = step_size

       
        self.var_idx = {var: i for i, var in enumerate(variables)}
        self.num_vars = len(variables)

        
        self.direct_effects = []
        self.indirect_effects = []
        self.total_effects = []
        self.state_representations = []
        self.rankings = []

    def process_time_steps(self):
        
        for t in range(self.time_steps):
            #print(f"\n==== Processing Time Step {t + 1} ====")

            
            start_idx = t * self.step_size
            end_idx = (t + 1) * self.step_size
            data_t = self.data.iloc[start_idx:end_idx].copy()

            
            self._process_single_step(data_t, t)

    def _process_single_step(self, data_t: pd.DataFrame, t: int):
        """处理单个时间步的内部方法"""
        
        model = BayesianNetwork(self.edges)
        model.fit(data_t, estimator=MaximumLikelihoodEstimator)

        
        gamma = self._calculate_direct_effects(data_t)
        indirect = self._calculate_indirect_effects(gamma)
        total = gamma + indirect

        
        mu_n = data_t.mean(axis=0).values
        v_matrix = np.cov(data_t.T)  
        phi_matrix = total

        s_i = self._compute_state_representation(mu_n, v_matrix, phi_matrix)

        
        self.direct_effects.append(gamma)
        self.indirect_effects.append(indirect)
        self.total_effects.append(total)
        self.state_representations.append(s_i)

        
        self.rankings.append(np.argsort(-s_i) + 1)  

    def _calculate_direct_effects(self, data_t: pd.DataFrame) -> np.ndarray:
        
        gamma = np.zeros((self.num_vars, self.num_vars))

        for parent, child in self.edges:
           
            p_idx = self.var_idx[parent]
            c_idx = self.var_idx[child]

            
            X = data_t[[parent]].values.reshape(-1, 1)
            y = data_t[child].values.reshape(-1, 1)

            reg = LinearRegression().fit(X, y)
            gamma[p_idx, c_idx] = reg.coef_[0][0]

        return gamma

    def _calculate_indirect_effects(self, gamma: np.ndarray) -> np.ndarray:
        
        indirect = np.zeros_like(gamma)
        n = self.num_vars

        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                
                for k in range(n):
                    indirect[i, j] += gamma[i, k] * gamma[k, j]

                
                for k in range(n):
                    for m in range(n):
                        indirect[i, j] += gamma[i, k] * gamma[k, m] * gamma[m, j]

        return indirect

    def _compute_state_representation(self, mu_n: np.ndarray,
                                      v_matrix: np.ndarray,
                                      phi_matrix: np.ndarray) -> np.ndarray:
        
        lambda_i = np.zeros(self.num_vars)
        phi_i = np.zeros(self.num_vars)

        for i in range(self.num_vars):
            
            lambda_term = mu_n[i] ** 2 * v_matrix[i, i]
            cross_term = 0
            for j in range(self.num_vars):
                if j != i:
                    cross_term += mu_n[i] * v_matrix[i, j] * mu_n[j]
            lambda_i[i] = lambda_term + cross_term

           
            phi_term = mu_n[i] ** 2 * phi_matrix[i, i]
            phi_cross = 0
            for j in range(self.num_vars):
                if j != i:
                    phi_cross += mu_n[i] * phi_matrix[i, j] * mu_n[j]
            phi_i[i] = phi_term + phi_cross

        return lambda_i


