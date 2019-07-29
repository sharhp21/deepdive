from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris = load_iris()
kf_data = iris.keys()
# print("<< kf_data <<")
# print(kf_data)

kf_data = iris.data
kf_label = iris.target
kf_columns = iris.feature_names

# alt+shift+E
kf_data = pd.DataFrame(kf_data, columns=kf_columns)
# print("<< kf_label <<")
# print(kf_label)
# print(pd.value_counts(kf_label))
# print(kf_label.sum())
# print(kf_label.dtype)

def Kfold():
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, random_state=0)

    for i, (train_idx,valid_idx) in enumerate(kf.split(kf_data.values, kf_label)):
        train_data, train_label = kf_data.values[train_idx, :], kf_label[train_idx]
        valid_data, valid_label = kf_data.values[valid_idx, :], kf_label[valid_idx]
        print("{} Fold train label\n{}".format(i, train_label))
        print("{} Fold valid label\n{}".format(i, valid_label))

def Stratified_KFold():
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    kf = StratifiedKFold(n_splits=5, random_state=0)
    val_scores = list()

    for i, (train_idx, valid_idx) in enumerate(kf.split(kf_data.values, kf_label)):
        train_data, train_label = kf_data.values[train_idx, :], kf_label[train_idx]
        valid_data, valid_label = kf_data.values[valid_idx, :], kf_label[valid_idx]
        # print("{} Fold train label\n{}".format(i, train_label))
        # print("{} Fold valid label\n{}".format(i, valid_label))

        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=2019)
        # training
        clf.fit(train_data, train_label)
        train_acc = clf.score(train_data, train_label) * 100
        valid_acc = clf.score(valid_data, valid_label) * 100
        val_scores.append(valid_acc)
        print("{}fold, train Accuracy: {:.2f}%, validation Accuracy: {:.2f}%".format(i, train_acc, valid_acc))
        # val_scores.append(valid_acc)
    
    # Mean validation Score
    print("Cross Validation Score: {:.2f}%".format(np.mean(val_scores)))

if __name__ == "__main__":
    # Kfold()
    Stratified_KFold()