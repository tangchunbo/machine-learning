import numpy as np
from metrics import accuracy_score


class LogisticRegression:
    def __init__(self):
        """
        coef_ : 模型系数 （Theta1 - ThetaN）
        intercepti_: 模型截距 Theta0
        _theta: Theta
        :return:
        """
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def _sigmoid(self, t):
        return 1. / (1. + np.exp(-t))

    def fit(self, X_train, y_train, eta=0.01, n_iters=1e4):
        # 损失函数
        def J(theta, X_b, y):
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                return - np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(y)
            except Exception:
                return float('inf')

        # 损失函数
        def dJ(theta, X_b, y):
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(X_b)

        def gradient_descent(X_b, y, init_theta, epsilon=1e-8):
            theta = init_theta
            i_iter = 0

            while i_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                i_iter += 1
                if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
                    break
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict_proba(self, X_predict):
        """给定待预测数据集X_predict,返回表示X_predict的【结果概率向量】"""
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return self._sigmoid(X_b.dot(self._theta))

    def predict(self, X_predict):
        """给定待预测数据集X_predict,返回表示X_predict的【结果向量】"""
        prob = self.predict_proba(X_predict)
        return np.array(prob >= 0.5, dtype=int)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"
