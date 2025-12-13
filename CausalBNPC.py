import numpy as np
import pandas as pd
from pgmpy.estimators import PC
from sklearn.linear_model import LinearRegression
np.random.seed(128)

def compute_total_effect_matrix(data: pd.DataFrame, significance_level: float = 0.1,
                                assumed_edges: list = None) -> np.ndarray:

    if assumed_edges is not None and len(assumed_edges) > 0:
        edges = assumed_edges
        #print("Using assumed edges:", edges)
    else:
        
        pc = PC(data)
        model = pc.estimate(significance_level=significance_level)
        edges = list(model.edges())
        #print("Learned edges:", edges)

    if len(edges) == 0:
        print("Warning: No edges detected; please check data or provide assumed_edges.")

    
    variables = list(data.columns)
    var_idx = {var: i for i, var in enumerate(variables)}
    num_vars = len(variables)

    
    gamma = np.zeros((num_vars, num_vars))
    for parent, child in edges:
        p_idx = var_idx[parent]
        c_idx = var_idx[child]
        X = data[[parent]].values.reshape(-1, 1)
        y = data[child].values.reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        coef = reg.coef_[0][0]
        gamma[p_idx, c_idx] = coef
        #print(f"Direct effect from {parent} to {child}: {coef}")

    
    indirect = np.zeros((num_vars, num_vars))

    for i in range(num_vars):
        for j in range(num_vars):
            if i == j:
                continue
            for k in range(num_vars):
                indirect[i, j] += gamma[i, k] * gamma[k, j]

    for i in range(num_vars):
        for j in range(num_vars):
            if i == j:
                continue
            for k in range(num_vars):
                for m in range(num_vars):
                    indirect[i, j] += gamma[i, k] * gamma[k, m] * gamma[m, j]

    
    total_effect = gamma + indirect
    np.fill_diagonal(total_effect, 1)
    return np.abs(total_effect)


def compute_causal_statistic(mu: np.ndarray, total_effect: np.ndarray) -> np.ndarray:

    p = len(mu)
    phi = np.zeros(p)
    for i in range(p):
        
        phi[i] = mu[i] ** 2
        
        for j in range(p):
            if i != j:
                phi[i] += mu[i] * total_effect[i, j] * mu[j]
    return phi




