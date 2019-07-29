import numpy as np

# npa = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(npa.size)
# print(npa.shape)
# print(len(npa))
# print(npa.ndim)

# npl = np.array([1, 100, 42, 42, 42, 6, 7])
# print(npl.size)
# print(len(npl))
# print(npl.shape)
# print(npa.ndim)

# a = np.zeros(3)
# print(a.ndim)
# print(a.shape)
# print(a)

# print(np.eye(2, k = 1, dtype = int))
# print(np.eye(3))
# print(np.eye(3, k=1))
# print(np.eye(3, k=-1))

# print(np.identity(5))

# x = np.array([1, 5, 2])
# y = np.array([7, 4, 1])
# print(x + y)
# print(x * y)
# print(x - y)
# print(x / y)
# print(x % y)

# bb = np.array([1, 2, 3])
# cc = np.array([-7, 8, 9])
# print(np.dot(bb, cc))

# xs = np.array(((2, 3), (3, 5)))
# ys = np.array(((1, 2), (5, -1)))
# print(np.dot(xs, ys), type(np.dot(xs, ys)))

# l33 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# np33 = np.array(l33, dtype = int)

# print(np33.shape)
# print(np33.ndim)
# print(np33)
# print("first row : ", np33[0])
# print("first column : ",np33[:,0])
# print(np33[:2,1:])

# arr = np.array([9, 18, 29, 39, 49])

# print(" index ")
# print(arr.argmax())
# print(arr.argmin())

# print(" value ")
# print(arr[np.argmax(arr)])
# print(arr[np.argmin(arr)])

# a = np.arange(6)
# b = np.arange(6).reshape(-1, 3)
# a[5] = 100

# print(a)
# print(b)
# print(a[np.argmax(a)])

# print(np.argmax(b, axis = 0)) # 열
# print(np.argmax(b, axis = 1)) # 행

# a = np.random.rand(3, 2)
# print(a)

# b = np.random.rand(3, 3, 3)
# print(b)

# outcome = np.random.randint(1, 7, size = 10)
# print(outcome)
# print(type(outcome))
# print(len(outcome))

# print(np.random.randint(2, size=10))
# print(np.random.randint(1, size=10))
# print(np.random.randint(5, size=(2,4)))

import matplotlib.pyplot as plt

# a = np.random.randn(3,2)
# print(a)

# b = np.random.randn(3, 3, 3)
# print(b)

# plt.plot(a)
# plt.show()

# arr = np.arange(10)
# print(arr)
# np.random.shuffle(arr)
# print(arr)

# arr2 = np.arange(9).reshape((3, 3))
# print(arr2)
# np.random.shuffle(arr2)
# print(arr2)

#기초통계함수
from scipy.stats import mode

x = np.array([-2.1, -1, 1, 1, 4.3])
print(np.mean(x))
print(np.median(x))
print(mode(x)) #최빈값

import statistics as sta

x_m = np.mean(x)
x_a = x - x_m
x_p = np.power(x_a, 2)

print(x_a)
print(x_p)
print(" Variance x ") #분산 : 편차 제곱의 평균
print(np.var(x))
print(sta.pvariance(x))
print(sta.variance(x))

print(np.std(x)) #표준편차 : 분산의 제곱근
print(sta.pstdev(x))
print(sta.stdev(x))