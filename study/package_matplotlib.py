import matplotlib.pyplot as plt
import numpy as np

# plt.plot([1,2,3,4], [1,4,9,16]) #그래프 그리기
# plt.show() #그래프 출력

# X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
# C, S = np.cos(X), np.sin(X)

# pt1 = plt.plot(X, C)
# pt2 = plt.plot(X, S)
# print(pt1, pt2)
# plt.show()

# years = [x for x in range(1950, 2011, 10)]
# gdp = [y for y in np.random.randint(300, 10000, size=7)]

# plt.plot(years, gdp, color='c')
# print(gdp)
# plt.show()

#선 스타일
# t = np.arange(0, 5, 0.2)
# a = plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
# plt.show()
# print(a)

# x = [1,2,3,4,5]
# y = [1,4,9,16,25]
# plt.plot(x,y, linewidth=3.0)
# plt.show()

# years = [x for x in range(1950, 2011, 10)]
# gdp = [y for y in np.random.randint(300, 10000, size = 7)]
# plt.plot(years, gdp, marker='p', markersize=6, markeredgewidth=1, markeredgecolor='red', markerfacecolor='green')
# plt.show()

# plt.plot([1,2,3,4], 'ro')
# plt.show()

# a = b = np.arange(0, 3, .02)
# c = np.exp(a)
# d = c[::-1] #거꾸로 집어넣기

# plt.plot(a, c, 'k--', label = "Model length")
# plt.plot(a, d, 'k:', label = 'Data length')
# plt.plot(a, c+d, 'k', label = 'Total message length')
# plt.legend() # 범례 표시
# plt.show()

# a = plt.plot([1,2,3,4],[1,4,9,16],'ro')
# b = plt.axis([0,6,0,20])
# print(a)
# print(type(a), b)
# plt.show()

# N = 5

# menMeans = (20, 35, 30, 35, 27)
# width = 0.01
# ind = np.arange(N)
# print(ind)
# plt.bar(ind, menMeans)  # bar?
# plt.title('Scores by group and gender')
# plt.ylabel("Scores")
# plt.xlabel("The number of people")
# plt.xticks(ind + width/2., ('G1', 'G2', 'G3', 'G4', 'G5')) # 세부값 부여
# plt.yticks(np.arange(0, 81, 10))
# plt.legend(('Men')) #범례표시
# plt.show()

# x = [1,2,3,4,5]
# y = [1,4,9,16,25]
# plt.plot(x, y)
# plt.title('Plot of y sv x')
# plt.xlabel('x axis')
# plt.ylabel('y axis')
# plt.xlim(0.0, 7.0) #범위 부여
# plt.ylim(0.0, 30.0)
# plt.show()

# data = np.random.rand(10, 2)
# print(data)
# print(data.shape)

# plt.scatter(data[:,0], data[:,1])
# plt.show()

# x = np.random.rand(10)
# y = np.random.rand(10)
# z = np.sqrt(x**2 + y**2)

# plt.subplot(321)
# plt.scatter(x, y, s=80, c=z, marker='>')
# plt.show()

# N=5
# menMeans = (20, 35, 30, 35, 27)
# width = 0.01
# ind = np.arange(N)
# print(ind)
# plt.bar(ind, menMeans) #x 데이터, y 데이터
# plt.show()

# w_pop = np.array([5, 30, 45, 40], dtype = np.float32)
# m_pop = np.array([4, 28, 40, 35], dtype = np.float32)
# x = np.arange(4)
# a = plt.barh(x, w_pop, color='r') #수평막대 그래프
# b = plt.barh(x, -m_pop) #다중 수평막대 그래프 - 기호 주의

# print(a)
# print(b)
# plt.show()

# data = [5, 10, 30, 20, 7, 8, 10, 10]
# plt.pie(data)
# plt.show()

data = np.random.normal(5.0, 3.0, 1000)
plt.hist(data, bins=15, facecolor='red', alpha=0.4)
plt.xlabel('data')
print(data)
plt.show()