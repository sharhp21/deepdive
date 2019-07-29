#필요한 패키지 import
import os
from os.path import join
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

#데이터 확보
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.DESCR) #주택 가격에 대한 설명
data = boston.data
label = boston.target
print(label)
columns = boston.feature_names

#데이터 프레임 변화
data = pd.DataFrame(data, columns=columns)
print(data.head())
print(data.shape)
print(data.describe())
print(data.info()) #자료형

#데이터 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=2019)

#모델불러오기
from sklearn.linear_model import LinearRegression
sim_lr = LinearRegression()
sim_lr.fit(x_train['RM'].values.reshape((-1,1)), y_train)
y_pred = sim_lr.predict(x_test['RM'].values.reshape((-1,1)))

#결과 불러오기, 그림그리기
from data import bostonData
from sklearn.metrics import r2_score
print('단순 선형 회귀{:.4f}'.format(r2_score(bostonData.y_test,y_pred)))
print('단순 선형 회귀, 계수(w):{:.4f}, 절편(b):{:.4f}'.format(sim_lr.coef_[0], sim_lr.intercept_))
plt.scatter(x_test['RM'], y_test, s=10, c='black')
plt.plot(x_test['RM'], y_pred, c='red')
plt.legend(['Regression line', 'x_test'], loc='upper left')
plt.show()

##multiple linear regression
mul_lr=LinearRegression()
mul_lr.fit(bostonData.x_train.bostonData.y_train) #학습
mul_y_pred=mul_lr.predict(bostonData.x_test) #예측
print("다중 선형회귀 계수:{}. 절편:{.4f}".format(mul_lr.coef_, mul_lr.intercept_))
print("다중 선형 회귀, R2:{:.4f}".format(r2_score(bostonData.y_test,y_pred)))
