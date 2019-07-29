import os
from os.path import join
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston


boston=load_boston()
# print(boston.DESCR)
data=boston.data
label=boston.target
# print(label)
columns=boston.feature_names

data=pd.DataFrame(data,columns=columns)
# print(data)
# print(data.head())
# print(data.shape)
# print(data.describe())
# print(data.info())

x_train,x_test,y_train,y_test=train_test_split(data,label,test_size=0.2,random_state=2019)
sim_lr=LinearRegression()
sim_lr.fit(x_train['RM'].values.reshape((-1,1)),y_train)
y_predict= sim_lr.predict(x_test['RM'].values.reshape((-1,1)))

print('단순 선형회귀 r2 {:.4f}'.format(r2_score(y_test,y_predict)))
print('계수 w {:.4f}, 절편 b {:.4f}'.format(sim_lr.coef_[0],sim_lr.intercept_))

plt.scatter(x_test['RM'],y_test,s=10,c='black')
plt.plot(x_test['RM'],y_predict,c='red')
plt.show()

mul_lr=LinearRegression()
mul_lr.fit(x_train.values,y_train)
y_pred=mul_lr.predict(x_test.values)

print('다중 선형회귀 계수 :{}  절편 {:.4f}'.format(mul_lr.coef_,mul_lr.intercept_))
print('다중 선형 회귀, R2 : {:.4f}',format(r2_score(y_test,y_pred)))