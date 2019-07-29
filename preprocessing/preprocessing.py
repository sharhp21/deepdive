import pandas as pd
import numpy as np
from os.path import join
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# 데이터 불러오기
# data = pd.read_csv("D:/deepdive/data/abalone.csv")
# print(data.shape)
# print(data.describe())

# print(data.info())

# label=data['Sex']
# del data["Sex"]

def importData():
    abalone_path = join('D:/deepdive/data', 'abalone.txt')
    column_path = join('D:/deepdive/data', 'abalone_attributes.txt')
    # print(abalone_path)
    # print(column_path)
    abalone_columns = list()
    for l in open(column_path):
        abalone_columns.append(l.strip())
    # print(abalone_columns)
    data = pd.read_csv(abalone_path, header=None, names=abalone_columns)
    data.shape
    # print("describe=\n", data.describe())
    # data.info()
    # label=data['Sex']
    # del data["Sex"]
    return data
    
def mmscaler(data):
    #특징선언
    data = (data-np.min(data))/(np.max(data)-np.min(data)) #실제공식
    mscaler = MinMaxScaler()
    #특징찾기
    # print("data.head()=>", data)
    mscaler.fit(data)
    mMscaled_data = mscaler.transform(data)
    mMscaled_data_f = pd.DataFrame(mMscaled_data, columns = data.columns)
    # print("min_values=>", mMscaled_data.min())
    # print("max_values=>", mMscaled_data.max())
    print("data.mMscaled_data_f()=>", mMscaled_data_f.head())
    # print("data()=>", data)

def sdscaler(data):
    sdscaler = StandardScaler() #scaler 정의
    sdscaler.fit(data) #평균, 표준편차값 찾기
    sdscaler_data = sdscaler.transform(data) #변환
    sdscaler_pd = pd.DataFrame(sdscaler_data, columns=data.columns)
    print("sdscaler_data=>", sdscaler_pd.head())

def psampling(data):
    #print("sampling")
    #모델정의
    label = data['Sex']
    ros = RandomOverSampler(random_state=2019)
    rus = RandomUnderSampler(random_state=2019)

    #데이터에서 특징을 학습함과 동시에 데이터 샘플링

    #over 샘플링
    oversampled_data, oversampled_label = ros.fit_resample(data, label)
    oversampled_data = pd.DataFrame(oversampled_data, columns=data.columns)

    #under 샘플링
    undersampled_data, undersampled_label = rus.fit_resample(data, label)
    undersampled_data = pd.DataFrame(undersampled_data, columns = data.columns)

    print("원본데이터의 클래스비율\n{}".format(pd.get_dummies(label).sum()))
    print("\noversampled_data 클래스비율\n{}".format(pd.get_dummies(oversampled_label).sum()))
    print("\nundersampled_data 클래스비율\n{}".format(pd.get_dummies(undersampled_label).sum()))

if __name__ == "__main__":
    #  mmscaler(importData())
    #  sdscaler(importData())
    psampling(importData())