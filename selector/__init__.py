import numpy as np
from preprocessing import StandardScaler_, MinMaxScaler_ , SimplePolynomialFeatures_
from linear_model import LinearRegression_, RidgeRegression_

class Selector():
    def __init__(self, model="linear", scaler="standard", polynomial='1'): 
        self.model = self._select__model(model)
        self.scaler = self._select_scaler(scaler)
        self.polynomial = self._select__polynomial(polynomial)

    def _select__model(self, model):
        if model == 'linear':
            return LinearRegression_()
        else:
            return RidgeRegression_()

    def _select__polynomial(self, polynomial):
        return SimplePolynomialFeatures_(int(polynomial))

    
    def _select_scaler(self, scaler):
        if scaler == 'standard':
            return StandardScaler_()
        else:
            return MinMaxScaler_()

    def build_pipeline(self):
        return Pipeline([('polynomial', self.polynomial),
                        ('scaler', self.scaler),
                        ('linear', self.model)])
        
class Pipeline():
    def __init__(self, steps):
        self.steps = steps

    def _check_data(self,X):
        if X.shape[1] != self.dims[1]:
            print("Error! Data dimensions are not compatible with model dimensions.")
            exit()

    def fit(self, X, y=None):
        X_ = np.copy(X)
        self.dims = X_.shape 
        for (_, model) in self.steps[:-1]:
            model.fit(X_)
            X_ = model.transform(X_)
        _, final_model = self.steps[-1] 
        costs = final_model.fit(X_, y, True, X)
        return costs

    def predict(self, X):
        X_ = np.copy(X)
        self._check_data(X_)
        for _, model in self.steps[:-1]:
            X_ = model.transform(X)
        _, predictor = self.steps[-1]
        return predictor.predict(X_)

    def transform(self, X):
        X_ = np.copy(X)
        self._check_data(X_)
        for _, model in self.steps:
            X_ = model.transform(X)
        return X_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def fit_predict(self, X, y=None):
        self.fit(X,y)
        return self.predict(X)