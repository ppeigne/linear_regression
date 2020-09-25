import numpy as np

class StandardScaler():
    def __init__(self, all_alone=False):
        self.all_alone = all_alone
        self.mean = None
        self.std = None

    def _get_mean(self, df):
        if not self.all_alone:
            mean = df.mean(axis=0)
        else:
            len_df = df.shape[0]
            num_features = df.shape[1]
            mean = np.zeros((1, num_features))
            for i in range(len_df):
                mean += df[i]
            mean /= len_df
        return mean
        
    def _get_std(self, df):
        if not self.all_alone:
            std = df.std(axis=0)
        else:
            len_df = df.shape[0]
            num_features = df.shape[1]
            std = np.zeros((1, num_features))
            for i in range(len_df):
                std += (df[i] - self.mean)**2
            std /= len_df
            std = np.sqrt(std)
        return std

    def fit(self, df):
        self.mean = self._get_mean(df)
        self.std = self._get_std(df)

    def transform(self, df):
        return (df - self.mean) / self.std

X = np.array([[4, 1, 2, 2],
      [1, 3, 9, 3],
      [5, 7, 5, 1]])
#x = np.arange(15).reshape((3,5))
#x[0,0] = 20
#print(x)
sc = StandardScaler(True)
sc.fit(X)
x_ = sc.transform(X)
print("True")
print(x_)
sc = StandardScaler()
sc.fit(X)
x_ = sc.transform(X)
print("False")
print(x_)

