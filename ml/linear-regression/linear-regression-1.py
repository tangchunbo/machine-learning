import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 3, 2, 3, 5])

plt.scatter(x, y)
plt.axis([0, 6, 0, 6])
plt.show()

x_mean = np.mean(x)
y_mean = np.mean(y)

num = 0.0
d = 0.0

for i, j in zip(x, y):
    num += (i - x_mean) * (j - y_mean)
    d += (i - x_mean) ** 2

a = num / d
b = y_mean - a * x_mean

print(a)
print(b)

y_hat = a * x + b

plt.scatter(x, y)
plt.plot(x, y_hat, color='r')
plt.axis([0, 6, 0, 6])
plt.show()