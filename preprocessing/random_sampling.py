import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def importdata():
    #데이터 불러오기
    data = pd.read_csv("D:/deepdive/data/abalone.csv")
    label = data["Sex"]
    #del data["Sex"]
    return data

def sdscalar(data):
    #scaling
    sdscalar = StandardScaler()
    sdscalar.fit(data) #평균, 표준편차값 찾기
    sdscalar_data = sdscalar.transform(data) #변환
    sdscalar_pd = pd.DataFrame(sdscalar_data, columns=data.columns)
    return sdscalar_pd

def psampling(data):
    label = data["Sex"]
    del data["Sex"]
    #성능 비교를 위한 test set 설정
    X_train, X_test, Y_train, Y_test = train_test_split(sdscalar(data), label, test_size=0.1, shuffle=True, random_state=5)

    #Random Sampling
    ros = RandomOverSampler(random_state=2019)
    rus = RandomUnderSampler(random_state=2019)
    oversampled_data, oversampled_label = ros.fit_resample(X_train, Y_train)
    undersampled_data, undersampled_label = rus.fit_resample(X_train, Y_train)
    oversampled_data = pd.DataFrame(oversampled_data, columns=data.columns)
    undersampled_data = pd.DataFrame(undersampled_data, columns=data.columns)
    storage = [X_train, X_test, Y_train, Y_test, oversampled_data, oversampled_label, undersampled_data, undersampled_label]
    return storage

    print("원본데이터의 클래스비율\n{}".format(pd.get_dummies(label).sum()))
    print("\noversampled_data 클래스비율\n{}".format(pd.get_dummies(oversampled_label).sum()))
    print("\nundersampled_data 클래스비율\n{}".format(pd.get_dummies(undersampled_label).sum()))

#성능비교
def train_and_test(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    accuracy = round(accuracy_score(Y_test, pred) * 100, 2)
    print("Accuracy : ", accuracy, "%")

if __name__ == '__main__':
    storage = psampling(importdata())
    print("original data")
    train_and_test(SVC(), storage[0], storage[2], storage[1], storage[3])
    print("oversample data")
    train_and_test(SVC(), storage[4], storage[5], storage[1], storage[3])
    print("undersample data")
    train_and_test(SVC(), storage[6], storage[7], storage[1], storage[3])
