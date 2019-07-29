import sys
sys.path.insert(0, "/deepdive")
from data import creditcard
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns; sns.set()
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
import numpy as np

def logisticR(x_train,y_train,x_test,y_test,disc):
    lr=LogisticRegression()
    lr.fit(x_train,y_train.ravel())
    y_test_pre = lr.predict(x_test)

    print(disc + "  accuracy_score   : {:.2f}%".format(accuracy_score(y_test,y_test_pre)*100))
    print(disc + "  recall_score   : {:.2f}%".format(recall_score(y_test, y_test_pre) * 100))
    print(disc + "  precision_score   : {:.2f}%".format(precision_score(y_test, y_test_pre) * 100))
    print(disc + "  roc_auc_score   : {:.2f}%".format(roc_auc_score(y_test, y_test_pre) * 100))

    cnf_matrix=confusion_matrix(y_test,y_test_pre)
    print(disc + " ===>\n",cnf_matrix) # the number of matrix
    print("cnf_matrix_test[0,0]>=",cnf_matrix[0,0])
    print("cnf_matrix_test[0,1]>=", cnf_matrix[0, 1])
    print("cnf_matrix_test[1,0]>=", cnf_matrix[1, 0])
    print("cnf_matrix_test[1,1]>=", cnf_matrix[1, 1])

    print(disc + "matrix_accuracy_score : ",(cnf_matrix[1,1]+cnf_matrix[0,0])/(cnf_matrix[1,0]+cnf_matrix[1,1]+cnf_matrix[0,1]+cnf_matrix[0,0])*100)
    print(disc + "matrix_recall_score : ",(cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])*100))

def rf(x_train,y_train,x_test,y_test,disc):
    rf=RandomForestClassifier()
    rf.fit(x_train,y_train.ravel())
    y_test_pre=rf.predict(x_test)

    cnf_matrix_rf=confusion_matrix(y_test,y_test_pre)
    print(disc + " matrix_accuracy_sciore : ",(cnf_matrix_rf[1,1]+cnf_matrix_rf[0,0])/(cnf_matrix_rf[1,0]+cnf_matrix_rf[1,1]+cnf_matrix_rf[0,1]+cnf_matrix_rf[0,0])*100)
    print(disc + " matrix_recall_score : ",(cnf_matrix_rf[1,1]/(cnf_matrix_rf[1,0]+cnf_matrix_rf[1,1])*100))

if __name__ == "__main__":
    x_train = creditcard.x_train
    y_train = creditcard.y_train
    x_test = creditcard.x_test
    y_test = creditcard.y_test

    x_smote = creditcard.x_train_res
    y_smote = creditcard.y_train_res

    logisticR(x_train,y_train,x_test,y_test,"smote전 +logisticR")
    logisticR(x_smote,y_smote,x_test,y_test,"smote후 +logisticR")
    rf(x_train,y_train,x_test,y_test,"smote전 +RF")
    rf(x_smote,y_smote,x_test,y_test,"smote후 +RF")

    # gridSearchCV()

    lr = LogisticRegression()
    parameters = {
        'C' : np.linspace(1, 10, 10),
        'penalty' : ['l1', 'l2']
    }
    clf = GridSearchCV(lr, parameters, cv=5, verbose=5, n_jobs=3) # n_jobs - using multi process
    print("<< clt - fit >>")
    clf.fit(x_smote, y_smote.ravel())
    print("<< Best params >>", clf.best_params_, clf.best_estimator_, clf.best_score_)