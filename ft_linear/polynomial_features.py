import numpy as np 

class PolynomialFeatures():
    def __init__(self, degree,
                interaction_only=False,
                include_bias=True):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
    
    def __rm_duplicate__(self, src, df):
        src_len = src.shape[1]
        df_len = df.shape[1]
        for i in range(src_len - rm):
            j = 0
            while j < df_len:
                if np.array_equal(src[:,i],df[:,j]):
                    df = np.delete(df, j, 1)
                    df_len = df.shape[1]
                j +=1
        return df

    def transform(self, df):
        tmp = np.copy(df)
        for i in range(1, self.degree):
            res = tmp
            for j in range(df.shape[1]):
                new_block = res[:,j].reshape(-1,1) * res[:,j:] # generate a new block of cols from res[:,j] times what remains in the df
                new_block = self.__rm_duplicate__(tmp, new_block) # remove diplicated cols from the new block 
                tmp = np.concatenate((tmp, new_block), axis=1)                
        return tmp

X = np.arange(6).reshape(3, 2)
print(X, end="\n\n")

p = PolynomialFeatures(4)
x_ = p.transform(X)
print(x_, end="\n\n")