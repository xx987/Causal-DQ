import numpy as np


class StateUpdater:
    def __init__(self, M, lam):

        self.M = M
        #self.sigma2 = sigma2
        #self.input_vector = input_vector
        self.lam = lam

        
        self.s = np.zeros(M)  
        self.w = np.zeros(M) 
        self.H = np.zeros(M, dtype=int) 
        self.mu = np.zeros(M)  
        self.state = None  

    def update(self, X, input_vector, sigma2, observed_idx):

        M = self.M

        
        self.w = (1 - self.lam) * self.w.copy()
        self.s = (1 - self.lam) * self.s.copy()

        
        for j in observed_idx:
            self.w[j] += 1.0 / sigma2[j] 
            self.s[j] += X[j] / sigma2[j] 

        
        #self.mu = np.zeros(M)
        for j in range(M):
            if self.w[j] > 0:
                self.mu[j] = self.s[j] / self.w[j]

            else:
                self.mu[j] = 0.0

        
        Lambda = self.mu ** 2 * self.w


        H_new = np.zeros(M, dtype=int)
        for j in range(M):
            if j in observed_idx:
                H_new[j] = 0
            else:
                H_new[j] = self.H[j] + 1
        self.H = H_new

        
        self.state = np.vstack([Lambda, input_vector, self.H])

        return self.state, self.s, self.w, self.H, self.mu


