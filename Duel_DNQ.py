import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import itertools
import random
from collections import deque
from State import StateUpdater as SG

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.advantage = nn.Linear(hidden, output_dim)
        # 价值分支（输出状态价值）
        self.value = nn.Linear(hidden, 1)

        self.out = nn.Linear(hidden, output_dim)



    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        adv = self.advantage(x)
        val = self.value(x)


        q = val + adv - adv.mean(dim=1, keepdim=True)

        return q


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action_idx, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action_idx),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done),
        )

    def __len__(self):
        return len(self.buffer)


class CausalEnv:
    def __init__(self, M=10, m=5):
        self.M = M
        self.m = m
        self.t = 0
        self.T = 31          # 漂移开始时间
        self.lam = 10
        self.p = 30
        self.ranking = np.random.permutation(self.M)
        # 初始化 StateGenerator（此处 lam 可单独设置，与环境 lam 不必一致）
        self.state_gen = SG(M, lam=0.08)
        #self.reset()

    def reset(self):
         self.t = 0

    def step(self,state, action_idxs, X , input_vector ,sigma2):#Input vector is the causal state here!
        self.t += 1
        #print(self.t,'self.t是多少')
        done = self.t >= 200
        self.w_s = np.clip(np.random.randn(3 * self.M), 0, 1)
        self.w_a = np.clip(np.random.randn(self.M,), 0, 1)

        # 计算 causal mask，用于奖励计算
        s_mask = np.zeros(self.M)
        s_mask[self.ranking.argsort()[:self.m]] = 1  # causal mask: M^{s→r}
        a_mask = np.zeros(self.M)
        #print(a_mask,'看看action idxs 是多少')
        a_mask[list(action_idxs)] = 1               # causal mask: M^{a→r}

        state_flat = state.flatten()
        s_mask_full = np.tile(s_mask, 3)
        #causal_s = s_mask_full * state_flat
        #causal_a = a_mask

        epsilon = np.random.normal(0, 0.1)
        C_t = 50#np.abs(self.w_s.dot(causal_s) + self.w_a.dot(causal_a) + epsilon)


        if self.t < self.T + self.lam:
            reward = -1.0 #if any(elem in action_idxs for elem in (0, 1, 2, 3)) else -100
        elif self.T + self.lam <= self.t <= self.T + self.p:
            #reward = max(C_t, np.exp((self.p - (self.t - self.T)) / 200000)) if any(elem in action_idxs for elem in (0, 1, 2, 3)) else -10
            #print(reward, 'reward22222222222222')
            #count = sum(1 for x in action_idxs if x in (0, 1, 2, 3))
            count = sum(1 for x in action_idxs if x in (0,1,2,3,4))#,6,7,8,9,10,11,12,13,14,15,16,17,18,19))
            reward = count if count > 0 else -50
            #reward = 3 if any(elem in action_idxs for elem in (0, 1, 2, 3)) else -10
            # print(reward, 'reward22222222222222')
        else:
            reward = 1
            #print(reward, 'reward333333333333333')

        # 构造上一轮动作的二值表示
        prev_action = np.zeros(self.M)
        prev_action[list(action_idxs)] = 1

        # 利用 StateGenerator 生成下一轮 state
        #E, Sigma, X = self.get_data()
        # 这里的 current_stat 可根据实际业务选择，这里用随机数模拟
        #current_stat = np.random.rand(self.M) * 2
        next_state = self.state_gen.update(X,input_vector,sigma2,action_idxs)[0]
        #print(reward, self.t,self.T,"RewardRewardRewardRewardReward CTCCCCCCCCCCC")
        return next_state, reward, done

# ---------------------------
# Double DQN Agent
# ---------------------------
class DoubleDQNAgent:
    def __init__(self, state_dim, action_list, alpha=0.1, gamma=0.99, lr=1e-3,tau_start=5,tau_end=0.01,tau_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_list = action_list
        #self.tau = tau      # Boltzmann 温度
        self.alpha = alpha  # 熵系数
        self.gamma = gamma

        self.q_net = QNetwork(input_dim=state_dim, output_dim=len(action_list)).to(self.device)
        self.target_net = QNetwork(input_dim=state_dim, output_dim=len(action_list)).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer()
        self.tau_start = tau_start
        self.tau = tau_start      # 当前温度初始化为起始值
        self.tau_end = tau_end
        self.tau_decay = tau_decay
        self.action_i = None

    def decay_tau(self):
        """指数衰减温度参数"""
        self.tau = max(self.tau_end, self.tau * self.tau_decay)
        return self.tau


    def select_action(self, state):
        # state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # with torch.no_grad():
        #     q_values = self.q_net(state).squeeze()
        #
        # probs = torch.softmax(q_values / self.tau, dim=0)
        # dist = torch.distributions.Categorical(probs)
        # action_idx = torch.multinomial(probs, num_samples=1, replacement=False).item()#dist.sample().item()
        #Before this line, old code"
        #action_indices = .tolist()
        #action_select =tuple(action_indices)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state).squeeze()
        #print(q_values,"q_valuesq_valuesq_valuesq_values")
        # 数值稳定版的Softmax计算
        #print(self.tau,"self.tau是多少self.tau是多少self.tau是多少self.tau是多少self.tau是多少")
        scaled_q = q_values / self.decay_tau()#self.tau
        scaled_q -= scaled_q.max()  # 防止数值溢出
        probs = torch.softmax(scaled_q, dim=0)
        #print(probs,"probsprobsprobsprobs")
        #action_idx = torch.multinomial(probs, num_samples=1).item()
        action_idx = torch.multinomial(probs, num_samples=5).tolist()
        self.action_i = action_idx
        #print(action_idx,"action_idxaction_idxaction_idxaction_idx")
        return action_idx, action_idx#self.action_list[action_idx]



        #print(q_values,"q_valuesq_valuesq_valuesq_valuesq_valuesq_values")
        #print()
        #return action_idx, self.action_list[action_idx]
        #return self.action_list.index(action_select), action_select

    def compute_entropy(self, q_values):
        # probs = torch.softmax(q_values / self.decay_tau(), dim=1)
        # log_probs = torch.log(probs + 1e-10)
        # entropy = -torch.sum(probs * log_probs, dim=1)
        # return entropy

        a_mask = torch.zeros_like(q_values)  # shape: (batch_size, num_actions)
        a_mask[:, self.action_i] = 1  # 只允许 action_i 中的动作为 1，其它为 0

        # 计算 masked softmax（即 C(a, s) * pi(a|s)）再归一化
        masked_logits = q_values / self.decay_tau()
        masked_logits[a_mask == 0] = 0#-1e10  # 给无效动作一个极小值，避免影响 softmax
        probs = torch.softmax(masked_logits, dim=1)

        # 计算熵
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=1)
        return entropy


    def update(self, batch_size=32):
        if len(self.replay) < batch_size:
            return

        state, action_idx, reward, next_state, done = self.replay.sample(batch_size)
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)
        action_idx = action_idx.to(self.device)

        q_values = self.q_net(state)
        #q_a = q_values.gather(1, action_idx.unsqueeze(1)).squeeze(1)
        q_a = q_values.gather(dim=1, index=action_idx)
        q_a = q_a.sum(dim=1)


        with torch.no_grad():
            next_q = self.q_net(next_state)
            next_actions = next_q.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_state)
            next_q_value = next_q_target.gather(1, next_actions).squeeze(1)

            entropy = self.compute_entropy(next_q)
            target = reward + self.gamma * (1 - done) * (next_q_value + self.alpha * entropy)

        loss = F.mse_loss(q_a, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())



