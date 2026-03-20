import numpy as np

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self,x,y):
        #converting to np arrays
        x = np.array(x)
        y = np.array(y)

        #adding coef_ with values '1'
        xb = np.c_[np.ones((x.shape[0],1)),x]

        #using the vector formula
        A = np.linalg.inv(xb.T @ xb) @ xb.T @ y

        #updating the values
        self.intercept_ = A[0]
        self.coef_ = A[1:]

    def predict(self,x):
        x = np.array(x)
        return x @ self.coef_ + self.intercept_ 