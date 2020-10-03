import numpy as np 
from sklearn.preprocessing import PolynomialFeatures

class PolynomialFeatures_():
    def __init__(self, degree,
                interaction_only=False,
                include_bias=True):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
    
    def _rm_duplicate(self, src, df):
        src_len = src.shape[1]
        df_len = df.shape[1]
        for i in range(src_len):
            j = 0
            while j < df_len:
                if np.allclose(src[:,i],df[:,j]):
                    df = np.delete(df, j, 1)
                    df_len = df.shape[1]
                j +=1
        return df

    def _generate_block(self, df, degree):
        tmp = np.copy(df)
        for _ in range(1, self.degree):
            res = tmp
            for j in range(df.shape[1]):
                new_block = res[:,j].reshape(-1,1) * res[:,j:] # generate a new block of cols from res[:,j] times what remains in the df
                new_block = self._rm_duplicate(tmp, new_block) # remove duplicated cols from the new block 
                tmp = np.concatenate((tmp, new_block), axis=1)    
        return tmp

    def transform(self, df):
        if self.interaction_only:
            tmp = self._generate_block(df, self.degree - 1)
            useless_vars = np.concatenate([df**i for i in range(2, self.degree)], axis=1)
            tmp = self._rm_duplicate(useless_vars, tmp)
        else:
            tmp = self._generate_block(df, self.degree)
        if self.include_bias:
            tmp = np.concatenate((np.ones((df.shape[0], 1)), tmp), axis=1)
        return tmp

X = np.arange(6).reshape(3, 2)
X_ = np.random.rand(6).reshape(3, 2)

p = PolynomialFeatures_(3, False, False)
x_ = p.transform(X)
print(x_.shape, end="\n\n")
p_ = PolynomialFeatures(3, False, False)
x__ = p_.fit_transform(X)
print(x__.shape, end="\n\n\n")
np.testing.assert_array_almost_equal(x_, x__, verbose=True)

p = PolynomialFeatures_(2, True, False)
x_ = p.transform(X)
print(x_.shape, end="\n\n")
print(x_)
p_ = PolynomialFeatures(2, True, False)
x__ = p_.fit_transform(X)
print(x__.shape, end="\n\n\n")
print(x__)

np.testing.assert_almost_equal(x_, x__)

p = PolynomialFeatures_(3, False, True)
x_ = p.transform(X)
print(x_.shape, end="\n\n")
p_ = PolynomialFeatures(3, False, True)
x__ = p_.fit_transform(X)
print(x__.shape, end="\n\n\n")
np.testing.assert_almost_equal(x_, x__)

p = PolynomialFeatures_(4)
x_ = p.transform(X)
print(x_.shape, end="\n\n")
p_ = PolynomialFeatures(4)
x__ = p_.fit_transform(X)
print(x__.shape, end="\n\n\n")

#np.testing.assert_almost_equal(x_, x__)"""