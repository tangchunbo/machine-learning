import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from SimpleLinearRegression1 import SimpleLinearRegression2
from model_selection import train_test_split
from math import sqrt

# boston 房产
boston = datasets.load_boston()
# 506 条记录
# 13个特征
# print(boston.DESCR)
# print(boston.feature_names)

# 取房间数量作为特征值

x = boston.data[:, 5]
y = boston.target

plt.scatter(x, y)
plt.show()

# 去除一些噪音
x = x[y < 50]
y = y[y < 50]

plt.scatter(x, y)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, seed=666)

print(x_train.shape)
reg = SimpleLinearRegression2()
reg.fit(x_train, y_train)

plt.scatter(x_train, y_train)
plt.plot(x_train, reg.predict(x_train), color='r')
plt.show()

y_predict = reg.predict(x_test)

# MSE

mse_test = np.sum((y_test - y_predict) ** 2) / len(y_test)
print(mse_test)

# RMSE
rmse_test = sqrt(mse_test)
print(rmse_test)

# MAE
mae_test = np.sum(np.absolute(y_test - y_predict)) / len(y_test)
print(mae_test)


# scikit-learn 中的指标
from sklearn.metrics import mean_squared_error, mean_absolute_error

sk_mse_test = mean_squared_error(y_test, y_predict)
print(sk_mse_test)

sk_mae_test = mean_absolute_error(y_predict, y_test)
print(sk_mae_test)