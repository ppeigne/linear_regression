import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self, alpha=.0001, n_cycle=100000):
        self.alpha = alpha
        self.n_cycles = n_cycle
        self.trained = False

    def add_intercept(self, X):
        return np.concatenate((np.ones(X.shape[0]).reshape(-1,1), X), axis=1)

    def normalize(self, X):
        X_ = np.copy(X.astype(float))
        if not self.trained:
            self.norms = [(X[:,i].std(), X[:,i].mean()) for i in range(X.shape[1])]    
        for i, (std, mean) in enumerate(self.norms):
            X_[:,i] = (X_[:,i] - mean) / std
        return X_

    def preprocess(self, X):
        X_ = np.copy(X) #.reshape(-1,1))
        return self.add_intercept(self.normalize(X_))

    def train(self, X, y, plot=True):
        #y = y.to_numpy().reshape(-1,1)
        X_ = self.preprocess(X)
        m, n = X_.shape
        self.theta = np.zeros((n, 1)) # .reshape(-1,1)
        for i in range(self.n_cycles + 1):
            self.theta -= self.alpha * (X_.T @ (X_ @ self.theta - y)/ m)
        self.trained = True 
        if plot:
            for i in range(1, n):
                print(i)
                plt.plot(X_[:,i], y, 'bo')
                plt.plot(X_[:,i], self.predict(X), 'ro')
            

    def predict(self, X):
        X_ = self.preprocess(X)
        if not self.trained:
            self.theta = self.theta = np.zeros((X.shape[1] + 1, 1))#.reshape(-1,1)
        return X_ @ self.theta

"""
LR = LinearRegression()
data = pd.read_csv("data.csv")
X = data["km"]
y = data["price"]
LR.train(X, y)
print(LR.theta)
print(LR.predict(X))
"""