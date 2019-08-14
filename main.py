from Classes.linear_regression import LinearRegression
from Classes.statistics import Statistics

import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    df = np.genfromtxt('data.csv', delimiter=',')
    df = df[1:]
    X, y = df[:, 0], df[:, 1]
   # print(X)
 #   X = np.stack((np.ones(X.shape[0]), X, X**2), axis=1)


    #X = np.stack((X, X**2), axis=1)

    #print(X.shape)

    #X = np.stack([np.ones(X.size), X], axis=1)

    s1 = Statistics(X)
    s1.print_stat()

    l1 = LinearRegression(X, y, np.zeros(2))
    res = l1.regularized_gradient_descent(0.5)
    print(l1.theta)
   # p = l1.predict([5000, 5000**2])
    p = l1.predict([5000])
    print(p)





    data = np.genfromtxt('ex1data2.txt', delimiter=',')
    #data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
    X = data[:, :2]
    y = data[:, 2]

    s2 = Statistics(X.T[0])
    s2.print_stat()

    s3 = Statistics(X.T[1])
    s3.print_stat()


    l2 = LinearRegression(X, y, np.zeros(3))
    res = l2.regularized_gradient_descent(0.1)
    print(l2.theta)
    p = l2.predict([1650, 3])
    #p = l1.predict([5000])
    print(p)
