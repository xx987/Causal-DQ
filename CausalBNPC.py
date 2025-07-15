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
        # 使用 PC 算法自动学习结构
        pc = PC(data)
        model = pc.estimate(significance_level=significance_level)
        edges = list(model.edges())
        #print("Learned edges:", edges)

    if len(edges) == 0:
        print("Warning: No edges detected; please check data or provide assumed_edges.")

    # 获取变量列表和建立变量索引映射
    variables = list(data.columns)
    var_idx = {var: i for i, var in enumerate(variables)}
    num_vars = len(variables)

    # 计算直接效应矩阵：对于每条边，通过线性回归估计效应系数
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

    # 计算间接效应（这里包括一阶和二阶间接效应）
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

    # 总效应矩阵 = 直接效应 + 间接效应
    total_effect = gamma + indirect
    np.fill_diagonal(total_effect, 1)
    return np.abs(total_effect)


def compute_causal_statistic(mu: np.ndarray, total_effect: np.ndarray) -> np.ndarray:

    p = len(mu)
    phi = np.zeros(p)
    for i in range(p):
        # 自身因果效应 mu_i^2 * total_effect[i,i] (这里 total_effect[i,i]==1)
        phi[i] = mu[i] ** 2
        # 累加其他变量的因果传播效应
        for j in range(p):
            if i != j:
                phi[i] += mu[i] * total_effect[i, j] * mu[j]
    return phi




#
# if __name__ == "__main__":
#     np.random.seed(0)
#     n_samples = 500
#
#     # 构造合成数据，使得变量之间存在明显的线性关系
#     data = pd.DataFrame({
#         'X1': np.random.randn(100),
#         'X2': np.random.randn(100),
#         'X3': np.random.randn(100),
#         'X4': np.random.randn(100)
#     })
#
#     print("Synthetic dataset (head):")
#     print(data.head())
#
#     # 假定的边结构：可以根据数据生成过程指定
#     assumed_edges = [
#         ('X1', 'X2'),
#         ('X1', 'X3'),
#         ('X2', 'X3'),
#         ('X2', 'X4'),
#         ('X3', 'X4')
#     ]
#
#     total_effect_matrix = compute_total_effect_matrix(data, assumed_edges=assumed_edges)
#     print("\nTotal Effect Matrix:")
#     print(total_effect_matrix)
