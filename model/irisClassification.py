import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.insert(0, "/deepdive/util")
from logfile import logger
import pandas as pd
import matplotlib.pyplot as plt

iris = load_iris()

##input data
#logger.debug(iris.DESCR)
logger.debug(iris.keys())
data = iris.data
label = iris.target
columns = iris.feature_names

data = pd.DataFrame(data, columns = columns)
#logger.debug(data.head())
#logger.debug(data.shape)
#logger.debug(data.describe())
#logger.debug(data.info())

##model check
#loger.debug("label=>", label)
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, shuffle=True, stratify=label, random_state=2019)

#model
lr = LogisticRegression()
logger.debug(y_train)
lr.fit(x_train, y_train) #training
lr_pred = lr.predict(x_test) #prediction
#logger.debug("lr_pred=>", lr_pred)
#logger.debug(lr.predict_proba(x_test))
logger.debug("logistic regression accuracy:{:.2f}%".format(accuracy_score(y_test, lr_pred) * 100))
logger.debug("logistic regression coef:{}, w:{}".format(lr.coef_, lr.intercept_))

