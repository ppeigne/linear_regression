import numpy as np 

class PolynomialFeatures():
    def __init__(self, degree,
                interaction_only=False,
                include_bias=True):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
    
    def rm_duplicate(self, src, df):
        src_len = src.shape[1]
        df_len = df.shape[1]
    #    print(src.shape)
    #    print(df.shape)
        rm = 0
        for i in range(src_len - rm):
            j = 0
            while j < df_len:
              #  print(j)
              #  print(df.shape)
              #  print(df_len, end="\n\n")
                if np.array_equal(src[:,i],df[:,j]):
                    df = np.delete(df, j, 1)
                    df_len = df.shape[1]
                j +=1
        return df

    def transform(self, df):
        res = np.copy(df)
        tmp = np.copy(res)
        for i in range(1, self.degree):
            res = np.copy(tmp)#, axis=1)
            for j in range(df.shape[1]):
                tmp = np.concatenate((tmp, self.rm_duplicate(tmp, res[:,j].reshape(-1,1) * res[:,j:])), axis=1)
                #tmp = self.rm_duplicate(res, res[:,j].reshape(-1,1) * res[:,j:])#), axis=1)
                
                print(i,j)
                print(tmp, end="\n\n")
        print(tmp)

X = np.arange(6).reshape(3, 2)
print(X, end="\n\n")

p = PolynomialFeatures(4)
x_ = p.transform(X)
print(x_, end="\n\n")