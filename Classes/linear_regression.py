import numpy as np


class LinearRegression:
    def __init__(self, X, y, theta, num_iter=400, alpha=0.1):
        self.X = np.column_stack((np.ones(X.shape[0]), X))
        self.m = self.X.shape[0]
        self.n = self.X.shape[1]
        self.sigma = self.sigma_value()
        self.mu = self.mu_value()
        self.normalized_X = self.mean_normalization(self.X)
        self.y = y
        self.theta = theta
        self.num_iter = num_iter
        self.alpha = alpha
        self.factor = self.alpha / self.m

    def cost_function(self, theta):
        error = self.normalized_X.dot(theta) - self.y
        return error.T.dot(error) / (2 * self.m)

    def mean_normalization(self, X):
        X_norm = X.copy()
        for i in range(1, self.n):
            X_norm.T[i] = (X_norm.T[i] - self.mu[i]) / self.sigma[i]
        return X_norm

    def mu_value(self):
        mu = np.ones(self.n)
        for i in range(1, self.n):
            mu[i] *= np.mean(self.X[:, i])
        return mu

    def sigma_value(self):
        sigma = np.ones(self.n)
        for i in range(1, self.n):
            sigma[i] *= np.std(self.X[:, i])
        return sigma

    def predict(self, x):
        x = np.array(x)
        x = np.concatenate((np.ones(1), x))
        x = self.mean_normalization(x)
        return np.dot(x, self.theta)

    def regularized_cost_function(self, theta, lambda_=0.1):
        error = self.normalized_X.dot(theta) - self.y
        return error.T.dot(error) / (2 * self.m) - lambda_ * theta[1:].dot(theta[1:])

    def regularized_gradient_descent(self, lambda_=0.1):
        th = self.theta.copy()
        tmp = self.theta.copy()
        rate = self.alpha / self.m
        for i in range(0, self.num_iter):
            tmp[0] -= rate * (self.normalized_X.dot(th) - self.y).dot(self.normalized_X[:, 0])
            tmp[1:] = th[1:] * (1 - rate * lambda_) - rate * self.normalized_X[:, 1:].T.dot(self.normalized_X.dot(th) - self.y)
            th = tmp
        self.theta = th
        return th

    def gradient_descent(self):
        th = self.theta.copy()
        tmp = self.theta.copy()
        rate = self.alpha / self.m
        for i in range(0, self.num_iter):
            tmp -= rate * self.normalized_X.T.dot(self.normalized_X.dot(th) - self.y)
            th = tmp
        self.theta = th
        return th

    def normal_equation(self):
        Xt = np.linalg.pinv(self.X)
        return (Xt.dot(self.X)).dot(Xt).dot(self.y)

