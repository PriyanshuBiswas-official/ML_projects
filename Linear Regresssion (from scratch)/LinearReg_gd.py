import numpy as np

class LinearRegressionGD:
    def __init__(self,learning_rate = 0.001, epochs = 1000):
        self.lr = learning_rate
        self.epoch = epochs
        self.coef_ = None
        self.intercept_ = 0.0
        self.loss_hist = []

    def fit(self, x,y):
        x = np.array(x)
        y = np.array(y)
        n_samples, n_features = x.shape

        #initializing weights to zero
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        for _ in range(self.epoch):
            y_pred = x @ self.coef_ + self.intercept_

            #computing loss function
            loss = np.mean((y_pred - y)**2)
            self.loss_hist.append(loss)

            #gradients
            error = y_pred - y
            grad_coef = (2/n_samples) * (x.T @ error)
            grad_intercept = (2/n_samples) * np.sum(error)

            #updating weights
            self.coef_ -= self.lr * grad_coef
            self.intercept_ -= self.lr * grad_intercept

    def predict(self, x):
        x = np.array(x)
        return x @ self.coef_ + self.intercept_
