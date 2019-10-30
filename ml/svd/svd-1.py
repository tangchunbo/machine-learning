import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import cv2

file_path = "./img/img.jpg"

image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
m, n = image.shape

# 原本L应该时一个对称矩阵，此时则将对称矩阵的对角线元素以一维数组形式输出
# P (m, m)
# L (m,)
# Q (n, n)
P, L, Q = np.linalg.svd(image)
print(P.shape)
print(Q.shape)
# L为800的一位数组，使用diag()输出以L为对角线的矩阵
tmp = np.diag(L)
if m < n:
    # 水平堆叠,目的就是为了将d扩充为mxn大小
    d = np.hstack((tmp, np.zeros((m, n - m))))
else:
    # 垂直堆叠
    d = np.vstack((tmp, np.zeros((m - n, n))))

k = 50

image2 = P[:, :k].dot(d[:k, :k]).dot(Q[:k, :])
io.imshow(np.uint8(image2))
plt.show()
