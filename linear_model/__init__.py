import numpy as np
import pandas as pd
from real_time_plots import init_real_time_plot, update_real_time_plot
import matplotlib.pyplot as plt

class LinearRegression_():
    def __init__(self, alpha=.001, n_cycle=100000):
        self.alpha = alpha
        self.n_cycles = n_cycle
        self.trained = False

    def cost_(self, X_, y):
        return (((X_ @ self.theta) - y) ** 2).sum()


    def add_intercept(self, X):
        return np.concatenate((np.ones(X.shape[0]).reshape(-1,1), X), axis=1)

    def fit(self, X, y, plot=True, X_original=None):
        X_ = self.add_intercept(X)
        y_ = np.array(y).reshape((-1,1)) 
        m, n = X_.shape
        self.theta = np.zeros((n, 1))
        self.cost = np.array([0, self.cost_(X_, y_)]).reshape(1, -1) 
        if plot:
            if X_original is None:
                X_original = X
            plots = init_real_time_plot(X_original, 
                                        y, 
                                        X_ @ self.theta, 
                                        self.n_cycles, 
                                        self.cost_(X_, y_)) 
        for i in range(self.n_cycles + 1):
            self.theta -= (self.alpha / m) * (X_.T @ (X_ @ self.theta - y_)) 
            self.cost = np.concatenate((self.cost, 
                                            np.array([i, self.cost_(X_, y_)]).reshape(1, -1)))
            if plot and i % 1000 == 0:
                update_real_time_plot(X_ @ self.theta, self.cost, plots)
        self.trained = True 
        return self.cost

    def predict(self, X):
        #check_data(X)
        X_ = self.add_intercept(X)
        if not self.trained:
            self.theta = self.theta = np.zeros((X.shape[1] + 1, 1))
        return X_ @ self.theta

class RidgeRegression_():
    def __init__(self, alpha=.001, n_cycle=100000, lambda_=0.01):
        self.alpha = alpha
        self.n_cycles = n_cycle
        self.lambda_ = lambda_
        self.trained = False


    def cost_(self, X_, y):
        return (((X_ @ self.theta) - y) ** 2).sum()

    def add_intercept(self, X):
        return np.concatenate((np.ones(X.shape[0]).reshape(-1,1), X), axis=1)

    def fit(self, X, y, plot=True, X_original=None):
        X_ = self.add_intercept(X)
        y_ = np.array(y).reshape((-1,1)) 
        m, n = X_.shape
        self.theta = np.zeros((n, 1))
        self.cost = np.array([0, self.cost_(X_, y_)]).reshape(1, -1) 
        if plot:
            if X_original is None:
                X_original = X
            plots = init_real_time_plot(X_original, 
                                        y, 
                                        X_ @ self.theta, 
                                        self.n_cycles, 
                                        self.cost_(X_, y_)) 
        for i in range(self.n_cycles + 1):
            tmp_theta = np.copy(self.theta)
            tmp_theta[0] = 0
            self.theta -= (self.alpha / m) * (X_.T @ (X_ @ self.theta - y_) + self.lambda_ * tmp_theta) 
            self.cost = np.concatenate((self.cost, 
                                            np.array([i, self.cost_(X_, y_)]).reshape(1, -1)))
            if plot and i % 1000 == 0:
                update_real_time_plot(X_ @ self.theta, self.cost, plots)
        self.trained = True 
        return self.cost

    def predict(self, X):
        X_ = self.add_intercept(X)
        if not self.trained:
            self.theta = self.theta = np.zeros((X.shape[1] + 1, 1))
        return X_ @ self.theta
