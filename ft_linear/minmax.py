import numpy as np

class MinMaxScaler():
    def __init__(self, all_alone=False):
        self.all_alone = all_alone
        self.min_ = None
        self.max_ = None

    def get_min(self, df):
        if not self.all_alone:
            min_ = df.min(axis=0)
        else:
            len_df = df.shape[0]
            num_features = df.shape[1]
            min_ = df[0]
            for j in range(num_features):
                for i in range(1, len_df):
                    if df[i,j] < min_[j]:
                        min_[j] = df[i,j]
        return min_
        
    def get_max(self, df):
        if not self.all_alone:
            max_ = df.max(axis=0)
        else:
            len_df = df.shape[0]
            num_features = df.shape[1]
            
            max_ = df[0]
            for j in range(num_features):
                for i in range(1, len_df):
                    if df[i,j] > max_[j]:
                        max_[j] = df[i,j]
        return max_

    def fit(self, df):
        self.min_ = self.get_min(np.copy(df))
        self.max_ = self.get_max(np.copy(df))

    def transform(self, df):
        return (df - self.min_) / (self.max_ - self.min_)

x = np.arange(15).reshape((3,5))
x[0,0] = 20

from sklearn.preprocessing import MinMaxScaler as mm
#x = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
print(x)
sc = MinMaxScaler(True)
sc.fit(x)
x_ = sc.transform(x)
print("True")
print(x_)
sc = MinMaxScaler()
sc.fit(x)
x_ = sc.transform(x)
print("False")
print(x_)
sc = mm()
sc.fit(x)
x_ = sc.transform(x)
print("Skl")
print(x_)