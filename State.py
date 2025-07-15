import numpy as np


class StateUpdater:
    def __init__(self, M, lam):

        self.M = M
        #self.sigma2 = sigma2
        #self.input_vector = input_vector
        self.lam = lam

        # 初始化各个中间变量，初始时 s, w, H, mu 均置为 0
        self.s = np.zeros(M)  # 累计分子，初始为 0
        self.w = np.zeros(M)  # 信息量或精度累积，初始为 0
        self.H = np.zeros(M, dtype=int)  # 未观测时间，初始均为 0
        self.mu = np.zeros(M)  # 贝叶斯均值，初始为 0
        self.state = None  # 状态矩阵，稍后构造

    def update(self, X, input_vector, sigma2, observed_idx):

        M = self.M

        # 1. 对所有数据流执行指数衰减更新
        self.w = (1 - self.lam) * self.w.copy()
        self.s = (1 - self.lam) * self.s.copy()

        # 2. 对当前被观测的数据流更新 s 和 w
        for j in observed_idx:
            self.w[j] += 1.0 / sigma2[j]  # sigma2[j]==1 时即加 1
            self.s[j] += X[j] / sigma2[j]  # sigma2[j]==1 时即加 X[j]

        # 3. 计算贝叶斯均值 mu：只有当 w > 0 时更新
        #self.mu = np.zeros(M)
        for j in range(M):
            if self.w[j] > 0:
                self.mu[j] = self.s[j] / self.w[j]

            else:
                self.mu[j] = 0.0

        # 4. 计算局部检测统计量 Λ(n)_i = mu[i]^2 * w[i]
        Lambda = self.mu ** 2 * self.w

        # 5. 更新未观测时间 H：
        # 如果当前 epoch 被观测，则 H 重置为 0，否则加 1
        H_new = np.zeros(M, dtype=int)
        for j in range(M):
            if j in observed_idx:
                H_new[j] = 0
            else:
                H_new[j] = self.H[j] + 1
        self.H = H_new

        # 6. 构造状态矩阵：三行分别为 Λ(n)_i、辅助 input_vector 和 H
        self.state = np.vstack([Lambda, input_vector, self.H])

        return self.state, self.s, self.w, self.H, self.mu


