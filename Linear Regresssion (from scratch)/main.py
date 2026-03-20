from LinearReg_closed import LinearRegression
from LinearReg_gd import LinearRegressionGD
from sklearn.datasets import  fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = fetch_california_housing()
# print(data.DESCR)
# print(data.feature_names)
# print(data.data)
# print(data.target)

x,y = data.data, data.target

# my model (closed form)
model = LinearRegression()
model.fit(x,y)
pred = model.predict(x)
print(pred)

print("R^2 value of Closed form Linear Regression model:",r2_score(y,pred))

# # my model (Gradient descent)
# modelgd = LinearRegressionGD()
# modelgd.fit(x,y)
# predgd = modelgd.predict(x)
# print(predgd)
# # print("R^2 value of Gradient descent Linear Regression model:",r2_score(y,predgd))

#sklearn model
modelsk = LinearRegression()
modelsk.fit(x,y)
predsk = modelsk.predict(x)
print(pred)
print("R^2 value for sklearn Linear Regression model:",r2_score(y,pred))

plt.figure(figsize=(8, 6))
plt.scatter(y, pred, color='steelblue', label='Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2, label='Perfect fit')

plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted (California Housing)')
plt.legend()
plt.tight_layout()
plt.show()