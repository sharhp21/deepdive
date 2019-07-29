from sklearn.datasets import load_wine

wine=load_wine()
# print(wine.DESCR)

# from util.logfile import logger
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

data=wine.data
label=wine.target
columns=wine.feature_names
data=pd.DataFrame(data,columns=columns)
# print(data)

x_train,x_test,y_train,y_test=train_test_split(data,label,stratify=label,random_state=0)
tree=DecisionTreeClassifier()
tree.fit(x_train,y_train)
score_tr=tree.score(x_train,y_train)
score_te = tree.score(x_test,y_test)
print('DT 훈련 정확도 {:.3f}'.format(score_tr))
print('DT테스트 세트 정확도 {:.3f}'.format(score_te))

tree1=DecisionTreeClassifier(max_depth=2)
tree1.fit(x_train,y_train)
score_tr1=tree1.score(x_train,y_train)
score_te1 = tree1.score(x_test,y_test)
print('DT 훈련 정확도 {:.3f}'.format(score_tr1))
print('DT테스트 세트 정확도 {:.3f}'.format(score_te1))

# tree2=DecisionTreeClassifier(max_depth=3)
# tree2.fit(x_train,y_train)
# score_tr2=tree2.score(x_train,y_train)
# score_te2 = tree2.score(x_test,y_test)
# print('DT 훈련 정확도 tree3 {:.3f}'.format(score_tr2))
# print('DT테스트 세트 정확도 tree3 {:.3f}'.format(score_te2))


import graphviz

from sklearn.tree import export_graphviz
export_graphviz(tree1,out_file='tree1.dot',class_names=wine.target_names,feature_names=wine.feature_names,impurity=False,filled=True)
with open('tree1.dot') as file_reader:
    dot_graph=file_reader.read()

dot=graphviz.Source(dot_graph)
dot.render(filename='tree1.png')

print("wine data.shape=> ",wine.data.shape)
n_feature = wine.data.shape[1]
print(n_feature)
idx=np.arange(n_feature)
print("idx=>",idx)
feature_imp=tree.feature_importances_ ### 트리의 depth에 따라 중요도가 달라지지만 가장 root node인 proline은 항상 가장중요함
plt.barh(idx,feature_imp,align='center')
plt.yticks(idx,wine.feature_names)
plt.xlabel('feature importance',size=15)
plt.ylabel('feture',size=15)

plt.show()