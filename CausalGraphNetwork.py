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

        # 创建变量索引映射
        self.var_idx = {var: i for i, var in enumerate(variables)}
        self.num_vars = len(variables)

        # 结果存储
        self.direct_effects = []
        self.indirect_effects = []
        self.total_effects = []
        self.state_representations = []
        self.rankings = []

    def process_time_steps(self):
        """处理所有时间步的主流程"""
        for t in range(self.time_steps):
            #print(f"\n==== Processing Time Step {t + 1} ====")

            # 获取当前时间步数据
            start_idx = t * self.step_size
            end_idx = (t + 1) * self.step_size
            data_t = self.data.iloc[start_idx:end_idx].copy()

            # 处理单个时间步
            self._process_single_step(data_t, t)

    def _process_single_step(self, data_t: pd.DataFrame, t: int):
        """处理单个时间步的内部方法"""
        # 1. 拟合贝叶斯网络
        model = BayesianNetwork(self.edges)
        model.fit(data_t, estimator=MaximumLikelihoodEstimator)

        # 2. 计算效应矩阵
        gamma = self._calculate_direct_effects(data_t)
        indirect = self._calculate_indirect_effects(gamma)
        total = gamma + indirect

        # 3. 计算状态表示
        mu_n = data_t.mean(axis=0).values
        v_matrix = np.cov(data_t.T)  # 使用协方差矩阵替代随机矩阵
        phi_matrix = total

        s_i = self._compute_state_representation(mu_n, v_matrix, phi_matrix)

        # 存储结果
        self.direct_effects.append(gamma)
        self.indirect_effects.append(indirect)
        self.total_effects.append(total)
        self.state_representations.append(s_i)

        # (可选) 添加排名逻辑
        self.rankings.append(np.argsort(-s_i) + 1)  # 降序排列的排名

    def _calculate_direct_effects(self, data_t: pd.DataFrame) -> np.ndarray:
        """计算直接效应矩阵"""
        gamma = np.zeros((self.num_vars, self.num_vars))

        for parent, child in self.edges:
            # 获取变量索引
            p_idx = self.var_idx[parent]
            c_idx = self.var_idx[child]

            # 拟合线性回归
            X = data_t[[parent]].values.reshape(-1, 1)
            y = data_t[child].values.reshape(-1, 1)

            reg = LinearRegression().fit(X, y)
            gamma[p_idx, c_idx] = reg.coef_[0][0]

        return gamma

    def _calculate_indirect_effects(self, gamma: np.ndarray) -> np.ndarray:
        """计算间接效应矩阵"""
        indirect = np.zeros_like(gamma)
        n = self.num_vars

        # 遍历所有变量对
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                # 一阶间接效应（i -> k -> j）
                for k in range(n):
                    indirect[i, j] += gamma[i, k] * gamma[k, j]

                # 二阶间接效应（i -> k -> m -> j）
                for k in range(n):
                    for m in range(n):
                        indirect[i, j] += gamma[i, k] * gamma[k, m] * gamma[m, j]

        return indirect

    def _compute_state_representation(self, mu_n: np.ndarray,
                                      v_matrix: np.ndarray,
                                      phi_matrix: np.ndarray) -> np.ndarray:
        """计算状态表示 s_i = phi_i + lambda_i"""
        lambda_i = np.zeros(self.num_vars)
        phi_i = np.zeros(self.num_vars)

        for i in range(self.num_vars):
            # 计算 lambda_i
            lambda_term = mu_n[i] ** 2 * v_matrix[i, i]
            cross_term = 0
            for j in range(self.num_vars):
                if j != i:
                    cross_term += mu_n[i] * v_matrix[i, j] * mu_n[j]
            lambda_i[i] = lambda_term + cross_term

            # 计算 phi_i
            phi_term = mu_n[i] ** 2 * phi_matrix[i, i]
            phi_cross = 0
            for j in range(self.num_vars):
                if j != i:
                    phi_cross += mu_n[i] * phi_matrix[i, j] * mu_n[j]
            phi_i[i] = phi_term + phi_cross

        return lambda_i


