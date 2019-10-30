import numpy as np


class SimpleLinearRegression1:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0
        for i, j in zip(x_train, y_train):
            num += (i - x_mean) * (j - y_mean)
            d += (i - x_mean) ** 2

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        """
        :param x_predict: np.array()
        :return:
        """
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression1()"


class SimpleLinearRegression2:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        """
        :param x_predict: np.array()
        :return:
        """
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression2()"


def test():
    x_train = np.array([1, 2, 3, 4, 5])
    y_train = np.array([1, 3, 2, 3, 5])
    simple1 = SimpleLinearRegression1()
    simple1.fit(x_train, y_train)
    print(simple1.predict(np.array([6, 7, 10])))

    simple2 = SimpleLinearRegression2()
    simple2.fit(x_train, y_train)
    print(simple2.predict(np.array([6, 7, 10])))



