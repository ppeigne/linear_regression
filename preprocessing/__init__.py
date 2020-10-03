import numpy as np

class MinMaxScaler_():
    def __init__(self, all_alone=False):
        self.all_alone = all_alone
        self.min_ = None
        self.max_ = None

    def _get_min(self, df):
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
        
    def _get_max(self, df):
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
        self.min_ = self._get_min(np.copy(df))
        self.max_ = self._get_max(np.copy(df))

    def transform(self, df):
        return (df - self.min_) / (self.max_ - self.min_)

    def fit_transform(self,df):
        self.fit(df)
        return self.transform(df)


class StandardScaler_():
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

    def fit_transform(self,df):
        self.fit(df)
        return self.transform(df)

class SimplePolynomialFeatures_():
    def __init__(self, degree=1):
        self.degree = degree
    
    def fit(self, df):
        pass

    def transform(self, df):
        tmp = np.concatenate([df ** i for i in range(1, self.degree + 1)], axis=1)
        return tmp

# import pandas as pd
# X = np.arange(12).reshape(6,2) 
# #X = pd.read_csv('spacecraft_data.csv')
# pol = SimplePolynomialFeatures_(3)
# print(X)
# print(pol.transform(X))

# class PolynomialFeatures_():
#     def __init__(self, degree,
#                 interaction_only=False,
#                 include_bias=True):
#         self.degree = degree
#         self.interaction_only = interaction_only
#         self.include_bias = include_bias
    
#     def _rm_duplicate(self, src, df):
#         src_len = src.shape[1]
#         df_len = df.shape[1]
#         for i in range(src_len):
#             j = 0
#             while j < df_len:
#                 if np.array_equal(src[:,i],df[:,j]):
#                     df = np.delete(df, j, 1)
#                     df_len = df.shape[1]
#                 j +=1
#         return df

#     def fit(self, df):
#         pass

#     def transform(self, df):
#         tmp = np.copy(df)
#         for _ in range(1, self.degree):
#             res = tmp
#             for j in range(df.shape[1]):
#                 new_block = res[:,j].reshape(-1,1) * res[:,j:] # generate a new block of cols from res[:,j] times what remains in the df
#                 new_block = self._rm_duplicate(tmp, new_block) # remove duplicated cols from the new block 
#                 tmp = np.concatenate((tmp, new_block), axis=1)                
#         if self.interaction_only:
#             useless_vars = np.concatenate([df**i for i in range(2, self.degree +1)], axis=1)
#             tmp = self._rm_duplicate(useless_vars, tmp)
#         if self.include_bias:
#             tmp = np.concatenate((np.ones((df.shape[0], 1)), tmp), axis=1)
#         return tmp