# linear_regression
A simple Python program doing linear and polynomial regressions over dataset. 
It handles univariate and multivariate regressions, using gradient descent or normal equation.

The program can display real-time visualization of the current state of the learning algorithm (actual prediction, cost function, confidence intervals).

## Screenshot

## Prerequisites
Numpy

## Features

### Regression types

|             | Linear Regression | Polynomial Regression |
| ----------  | :---------------: | :-------------------: |
| Univariate  |         ✔         |            ✔          |
| Multivariate |         ✔        |            ✔          |



### Real-time visualization

|                           | Data plot + prediction | Confidence Interval |Cost Function  |
| ------------------------  | :--------------------: | :-----------------: |:-------------------: |
| Univariate                |          ✔             |            ✔        | ✔
| Multivariate (2 features) |         ✔              | Numbers only        | ✔
| Multivariate (3+ features)|        ✖               | Numbers only        | ✔


Because of its instantaneous behaviour, normal equation is not compatible with real-time visualization.

## Usage

```
gradient_descent
regularized_gradient_descent
normal_equation

predict
```

