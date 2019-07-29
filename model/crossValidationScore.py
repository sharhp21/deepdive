from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate
from IPython.display import display
import pandas as pd

def train_test_split_():
    # make dataset
    x, y = make_blobs(random_state=0)
    print(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0)
    # training
    logreg = LogisticRegression().fit(x_train, y_train)
    # test
    print("테스트 세트 점수 : {:.2f}".format(logreg.score(x_test, y_test)))

def k_fold():
    iris = load_iris()
    kf_data = iris.data
    kf_label = iris.target
    kf_columns = iris.feature_names

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=2019)
    scores = cross_val_score(rf, kf_data, kf_label, cv=10)

    print(scores)
    print('rf k-fold CV score:{:2f}%'.format(scores.mean()))

def k_fold_validate():
    iris = load_iris()
    kf_data = iris.data
    kf_label = iris.target
    kf_columns = iris.feature_names

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=2019)
    scores = cross_validate(rf, kf_data, kf_label, cv=10, return_train_score=True)

    print("<< score >>")
    display(scores)
    res_df = pd.DataFrame(scores)
    print("<< res_df <<")
    display(res_df)
    print("평균 시간과 점수 : \n", res_df.mean())

def Stratified_KFold_ex():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import StratifiedKFold
    kfold = KFold(n_splits=5)
    stratifiedKfold = StratifiedKFold(n_splits=5)

    iris = load_iris()
    x, y = make_blobs(random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0)
    # training
    logreg = LogisticRegression().fit(x_train, y_train)
    print("교차 검증 점수(kfold) : \n", cross_val_score(logreg, iris.data, iris.target, cv=kfold))
    print("교차 검증 점수(stratifiedkfold) : \n", cross_val_score(logreg, iris.data, iris.target, cv=stratifiedKfold))

if __name__ == "__main__":
    # train_test_split_()
    # k_fold()
    # k_fold_validate()
    Stratified_KFold_ex()