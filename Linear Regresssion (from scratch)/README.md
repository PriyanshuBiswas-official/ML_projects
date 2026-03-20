# This is a project on implementing Linear Regression using just Numpy

There are two ways to implement linear regression:-

## 1. Linear Regression using Gradient Descent

Gradient Descent is an optimization algorithm used in linear regression to find the best-fit line for the data. It works by gradually adjusting the line’s slope and intercept to reduce the difference between actual and predicted values. This process helps the model make accurate predictions by minimizing errors step by step.

```
error = y_pred - y
grad_coef = (2/n_samples) * (x.T @ error)
grad_intercept = (2/n_samples) * np.sum(error)
```

## 2. Linear Regression using Vector form

Linear regression using a closed-form solution (Normal Equation) calculates the exact optimal parameters
that minimize the sum of squared errors directly in one step, without iterations. It uses matrix inversion:

```
A = np.linalg.inv(xb.T @ xb) @ xb.T @ y
```

providing the optimal slope and intercept immediately

### Note

Due to scaling issues of the california housing dataset, Linear Regression using gradient descent was not implemented in this project but it can be easily done by scaling the data. Large errors -> large gradients -> massive weight updates
