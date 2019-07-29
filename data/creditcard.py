import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

## data load
data=pd.read_csv('C:/Users/user/Desktop/creditcard.csv')
# print(data.head())
# print(data.columns)

## check data freq
# print(pd.value_counts(data['Class']))
pd.value_counts(data['Class']).plot.bar()
# plt.title('Fraud class histogram')
# plt.xlabel('Class')
# plt.ylabel('Frequncy')
# plt.show()

## amount stanardscaler preprocessing
sdscaler=StandardScaler()
data['normAmount']=sdscaler.fit_transform(data['Amount'].values.reshape(-1,1))
data=data.drop(['Time','Amount'],axis=1)
# print(data.head())

x=np.array(data.ix[:, data.columns !='Class'])
y=np.array(data.ix[:, data.columns =='Class'])
# print('Shape of X : {}'.format(x.shape))
# print('Shape of y : {}'.format(y.shape))

## divide train, test data

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=0)
# print("Number transaction X_train dataset : ", x_train.shape)
# print("Number transaction y_train dataset : ", y_train.shape)
# print("Number transaction X_test dataset : ", y_test.shape)
# print("Number transaction y_test dataset : ", y_test.shape)

## fit data imbalance
# print("Before OverSampling, counts of label '1' : {}".format(sum(y_train==1)))
# print("Before OverSampling, counts of label '0' : {}\n".format(sum(y_train==0)))
# print("y_train",y_train)
# print("y_train.ravel",y_train.ravel())

## smote
sm=SMOTE(random_state=2)
x_train_res,y_train_res=sm.fit_sample(x_train,y_train.ravel())

# print('After OverSampling, the shape of train_X : {}'.format(x_train_res.shape))
# print('After OverSampling, the shape of train_y : {}'.format(y_train_res.shape))

# print('After OverSampling, counts of y_train_res 1 : {}'.format(sum(y_train_res==1)))
# print('After OverSampling, counts of y_train_res 0 : {}'.format(sum(y_train_res==0)))

# print('After OverSampling, the shape of test_X : {}'.format(x_test.shape))
# print('After OverSampling, the shape of test_y : {}'.format(y_test.shape))

# print('before OverSampling, counts of label 1 : {}'.format(sum(y_test==1)))
# print('before OverSampling, counts of label 0 : {}'.format(sum(y_test==0)))