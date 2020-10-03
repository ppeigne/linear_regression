import matplotlib.pyplot as plt
import numpy as np

def real_time_fit(x, y, pred):
    plt.clf()
    plt.ylim(min(y)* 0.5, max(y) * 1.3)
    plt.xlim(min(x) * 1.3 , max(x)* 1.1)
    plt.scatter(x, y, color='orange')
    plt.plot(x, pred , color='lightblue')
    plt.pause(0.000001)
    plt.show()

def real_time_cost(n_max, cost):
    plt.clf()
    plt.plot(cost[:,0], cost[:,1], color='orange')
    plt.xlim(-10 , n_max)
    plt.pause(0.000000000001)
    plt.show()

def select_subplots_shape(n_features):
    if n_features > 16:
        print("Error! Too many features to produce real time plot!\nMax features: 8. Please select plot=False.")
        exit()
    dims = {
        2 : (1,2),
        3 : (1,3),
        4 : (2,2),
        5 : (2,3),
        6 : (2,3),
        7 : (2,4),
        8 : (2,4),
        9 : (3,3),
        10 : (3, 4),
        11 : (3, 4),
        12 : (3, 4),
        13 : (4, 4),
        14 : (4, 4),
        15 : (4, 4),
        16 : (3, 4)
    }
    return dims[n_features]

def init_real_time_plot(X, y, pred, n_max, cost):
    plt.ion()
    (i, j) = select_subplots_shape(X.shape[1] + 1)
    _, axes = plt.subplots(i,j)
    axes = axes.reshape(-1)
    plt.show()
    plt.subplots_adjust(hspace=.5)
    plt.subplots_adjust(wspace=.5)
    plots = []
    for n in range(len(axes)):
        if n < X.shape[1]:
            # fit plot
            axes[n].scatter(X.iloc[:,n], y, color='orange')
            axes[n].set_xlabel(X.columns[n])
            axes[n].set_ylabel(y.name)        
            fit_plot, = axes[n].plot(X.iloc[:,n], pred, '.', color='lightblue')
            axes[n].legend(['prediction', y.name])        
            plots.append(fit_plot)
        elif n == X.shape[1]:
            # cost plot
            cost_plot, = axes[n].plot(0, cost, color='orange')
            axes[n].set_xlim(-10 , n_max)
            axes[n].set_ylim(0, cost)
            axes[n].set_xlabel("iterations")
            axes[n].set_ylabel("cost (MSE)")
            plots.append(cost_plot)
        else:
            # blank plots
            axes[n].axis('off')            
    return plots

def update_real_time_plot(pred, cost, plots):
    for j in range(len(plots) - 1):
        plots[j].set_ydata(pred)
    plots[j + 1].set_data(cost[:,0], cost[:,1])
    plt.pause(.00001)
