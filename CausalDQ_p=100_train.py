from CausalGraphNetwork import CausalEffectAnalyzer as analyzer
from CausalBNPC import compute_total_effect_matrix as total_ec, compute_causal_statistic as c_causal_s
#from State import StateGenerator as StateG
from Duel_DNQ import DoubleDQNAgent, QNetwork, ReplayBuffer,CausalEnv
import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import random
from collections import deque

"--------------------------Create a dataset here-----------------------------"
# 定义参数
p = 100            # 变量维度（可改为100）
q = 20              # 偏移变量数（p=10时q=4；p=100时q=16）
delta = 1.0         # 均值偏移幅度
tau = 50            # 变化点位置（前tau个样本正常，之后偏移）
n_samples = 500    # 总样本量（时间序列长度）

# 生成协方差矩阵 Σ（对角线1，非对角线0.5）
cov_matrix = np.ones((p, p)) * 0.5
np.fill_diagonal(cov_matrix, 1)

# 生成均值偏移向量 μc（前q个变量为delta，其余为0）
mu_c = np.zeros(p)
mu_c[:q] = 0.5#1+np.random.rand(q)

# 生成数据集（前tau个样本正常，后n_samples-tau个样本偏移）
normal_data = np.random.multivariate_normal(
    mean=np.zeros(p),
    cov=cov_matrix,
    size=tau
)
shifted_data = np.random.multivariate_normal(
    mean=mu_c,
    cov=cov_matrix,
    size=n_samples - tau
)
dataset = np.vstack([normal_data, shifted_data])
"--------------------------Create a dataset here-----------------------------"



X_vars = [f'X{i}' for i in range(1, 101)]
np.random.seed(42)
data = pd.DataFrame(dataset,
                    columns= X_vars)    #['X1', 'X2', 'X3', 'X4', 'X5','X6', 'X7', 'X8', 'X9', 'X10'])

# 2. 定义网络结构
# example_edges = [('X1', 'X2'), ('X2', 'X3'),
#                  ('X3', 'X4'), ('X4', 'X5'),('X1', 'X5'),('X5', 'X6'),('X6', 'X7'),
#                  ('X7', 'X8'),('X8', 'X9'),('X9', 'X10')]


example_edges = [
    # 1. 链式连接
    *[(f"X{i}", f"X{i+1}") for i in range(1, 100)],

    # 2. 跨 5 跳的连接（X1-X6, X2-X7, …, X45-X50）
    *[(f"X{i}", f"X{i+5}") for i in range(1, 96)],

    # 3. 跨 10 跳的连接（X1-X11, X2-X12, …, X40-X50）
    *[(f"X{i}", f"X{i+10}") for i in range(1, 91)],

    # 4. 额外的随机/关键节点交叉连接
    ("X1",  "X10"),
    ("X5",  "X15"),
    ("X10", "X20"),
    ("X15", "X25"),
    ("X20", "X30"),
    ("X25", "X35"),
    ("X30", "X40"),
    ("X35", "X45"),
    ("X40", "X50"),
    ("X2",  "X14"),
    ("X7",  "X23"),
    ("X9",  "X33"),
    ("X12", "X27"),
    ("X18", "X31"),
    ("X22", "X47"),
    ("X29", "X49"),
    ("X8",  "X19"),
    ("X6",  "X28"),
    ("X11", "X39"),
    ("X16", "X44"),
]

causal_effect = total_ec(data=data, significance_level= 0.1,
                                assumed_edges=example_edges )

# analyzer =analyzer(
#     data=data,
#     edges=example_edges,
#     variables=  X_vars,  #['X1', 'X2', 'X3', 'X4', 'X5','X6', 'X7', 'X8', 'X9', 'X10'],
#     time_steps=100,
#     step_size=5
# )

# 4. 运行分析

import numpy as np

# 假设 StateGenerator 类已定义并导入

# 初始化参数
M = 100  # 总维度数
m = 22  # 每次选择的维度数
def get_all_actions(M, m):
    return list(itertools.combinations(range(M), m))

#ALL_ACTIONS = get_all_actions(M=10, m=5)
ALL_ACTIONS = np.arange(0, M).tolist()

# 创建 StateGenerator 实例
#state_generator =StateG(M)

# 初始化环境和代理
env = CausalEnv(M, m)
agent = DoubleDQNAgent(state_dim=3 * M, action_list=ALL_ACTIONS)

# 训练过程
# analyzer.process_time_steps()

"Causal Status, the second row of the status matrix"
# causal_status = analyzer.state_representations
# print(f"状态表示示例（第一步）: {analyzer.state_representations[0]}")

# start = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

start = np.array([[0]*100,
                  [0]*100,
                  [0]*100])



num_episodes = 500
reward_c=[]
for episode in range(num_episodes):
    state = start
    env.reset()
    # print(state, "STATESTATESTATESTATESTATE")
    total_reward = 0
    #print(episode,'episodeepisodeepisodeepisodeepisodeepisode')
    for step in range(300):
        # 这里的step应该设计为dataset 的time step即总共的time step（包括changing point 的T)
        # print(state, "S+TATESTATESTATESTATESTATE")


        action_idx, action = agent.select_action(state)
        print(action,"action")
        #print(action,"actionactionaction")

        "Make a selected mask for Z for computing E, mu, V"
        selected_mask = np.isin(np.arange(M), action).astype(int)
        Z = np.where(selected_mask == 1)[0].tolist()
        m = len(Z)
        # #E = np.zeros((M, M))
        # for row, idx in enumerate(Z):
        #     E[row, idx] = 1
        Sigma = np.eye(M)
        # #X = dataset[2, :]

        #Sigma = np.eye(10)
        #X = np.random.rand(10, 1)
        X = dataset[step, :]
        X = X.reshape(100, 1)

        #c_causal_s(mu = X, total_effect=causal_effect)






        #print(cc,'see each state')
        cc = c_causal_s(mu = X, total_effect=causal_effect) #analyzer.state_representations[0]
        sigma2 = np.ones(M)
        next_state, reward, done = env.step(state,action, X, cc,sigma2) #state, action_idxs, X , input_vector ,sigma2
        agent.replay.push(state, action_idx, reward, next_state, done)
        agent.update()
        state = next_state
        #print(state,'statestatestatestatestatestate')

        total_reward += reward
        #print(reward,'rewardrewardreward')

        if done:
            reward_c.append(total_reward)

            break
        #rint(step,"Stepstep")
        #print(state[2],"SEE last state")
        #print(reward,"rewardrewardrewardrewardrewardreward")
    reward_c.append(total_reward)
    #torch.save(agent.q_net.state_dict(), "dqn_cartpole.pth")
    #print(episode,'episodeepisodeepisodeepisode')
    #print(total_reward, 'totaltotaltotaltotaltotal')
    agent.update_target()


    #print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")
"After training--------------------------------------------------------"


torch.save(agent.q_net.state_dict(), 'q_net_100.pth')

agent.q_net.load_state_dict(torch.load('q_net_100.pth'))
agent.q_net.eval()  # 切换到评估模式














