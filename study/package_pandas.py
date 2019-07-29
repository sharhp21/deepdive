#DataFrame구조 : column별로 다른 데이터 타입을 가질 수 있음

import pandas as pd
import numpy as np

# obj1 = pd.DataFrame(data=np.arange(16).reshape(4,4), index=['a','b','c','d'], columns=['a','b','c','d'])
# print(obj1.index)
# print(obj1.columns)
# print(obj1.values)
# print(obj1.dtypes)

# d = [0,1,2,3,4,5,6,7,8,9]
# df = pd.DataFrame(d)
# df.columns = ['Rev']
# print(df)
# df['col'] = df['Rev'] #column 복사
# print(df)

# df['col'] = 5 #'col'열에 있는 value 모두 5로 변경
# print(df)
# df1 = df.drop("col", axis=1) #col 칼럼명에 해당하는 열 삭제
# print(df1)

# names = ['Bob', 'Jessica', 'Mary', 'John', 'Mel']
# births = [968, 155, 77, 578, 973]
# BabyDataSet = list(zip(names, births))
# print(BabyDataSet)
# print(type(BabyDataSet))

# df = pd.DataFrame(data=BabyDataSet, columns=['Names', 'Births'])
# print(df)
# print(df.shape)
# print(df.index) #행에대한 접근표시 확인
# print(df.columns) #열에 대한 접근표시 확인
# print(df.axes) #행과 열에 대한 축의 접근표시 확인

# #단일, 멀티 열 검색
# d = [0 , 1, 2, 3, 4, 5, 6, 7, 8, 9]
# df = pd.DataFrame(d)
# df.columns = ['Rev']
# df['col'] = df['Rev']
# i = ['a', 'b', 'c', 'd', 'f', 'e', 'g', 'h', 'i', 'j']
# df.index = i
# print(df['Rev'])
# print(df)
# print(df[['Rev', 'col']])

# #NULL여부 확인
# df = pd.DataFrame(np.arange(16).reshape(4,4), index=['a', 'b', 'c', 'd'], columns=['f', 'g', 'h', 'i'])
# print(df)
# df2 = df.reindex(['a','b','c','d','f','g','h'])
# print(df2)
# print(df2['f'].isnull())
# print(df2['f'].notnull())

# obj1 = pd.DataFrame(np.arange(16).reshape(4,4), index=['a','b','c','d'], columns=['a','b','c','d'])
# obj1.replace(to_replace=0, value=999, inplace=True)
# print(obj1)
# obj1.replace(to_replace=2, value=888, inplace=True)
# print(obj1)
# obj1['d'].replace(3, 777, inplace=True) #값으로 접근
# print(obj1)

# obj2 = pd.DataFrame(np.arange(16).reshape(4,4), index=['a','b','c','d'], columns=['a','b','c','d'])
# obj2.replace(to_replace=(0,2), value=999, inplace=True)
# print(obj2)
# obj2.replace(to_replace=[3,4,5], value=888, inplace=True)
# print(obj2)
# obj2['d'].replace((10,11), 777, inplace=True)
# print(obj2)

# d = {'one':[1,1,1,1,1], 'two':[2,2,2,2,2], 'letter':['a','a','b','b','c']}
# df1 = pd.DataFrame(d)
# print(df1)
# one = df1.groupby('letter')
# print(one)
# print(one.sum())

# letterone = df1.groupby(['letter', 'one']).sum()
# print(letterone)

# lettertwo = df1.groupby(['letter', 'one'], as_index=False).sum()
# print(lettertwo)
# print(lettertwo.index)

# df = pd.DataFrame(np.arange(30).reshape(5,6))
# print(df.describe()) #전체 통계 정보 조회

# print(df)
# print(df[0].describe())

# df = pd.DataFrame(np.arange(16).reshape(4,4), index=['a','b','c','d'], columns=['f','g','h','i'])

# print(df)
# print(df.sum(axis=0))
# print(df.mean(axis=0))
# print(df.std(axis=0))
# print(df.var(axis=0)) #행,열

# print(df.min(axis=0))
# print(df.max(axis=0))

# #DataFrame 연결
# df = pd.DataFrame(np.arange(16).reshape(4,4), index=['a','b','c','d'], columns=['f','g','h','i'])
# print(pd.concat([df, df]))
# print(pd.concat([df, df], axis=1))

# df1 = pd.DataFrame(np.arange(16, 32).reshape(4, 4), index=['a','b','c','d'], columns=['f','g','h','i'])
# print(pd.merge(df, df1))
# print(pd.merge(df, df, left_on='f', right_on='f'))

# raw_data = {
#     'subject_id': ['1', '2', '3', '4', '5'],
#     'first_name' : ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
#     'last_name' : ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']
# }

# df_a = pd.DataFrame(raw_data)

# raw_data = {
#     'subject_id': ['4', '5', '6', '7', '8'],
#     'first_name' : ['Billy', 'Brain', 'Bran', 'Bryce', 'Betty'],
#     'last_name' : ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']
# }

# df_b = pd.DataFrame(raw_data)

# print(df_a)
# print(df_b)
# print(pd.merge(df_a, df_b, on='subject_id'))
# print(pd.merge(df_a, df_b, on='subject_id', how='inner')) ## inner join


raw_data = {
    'subject_id': ['1', '2', '3', '4', '5'],
    'first_name' : ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
    'last_name' : ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']
}

df_a = pd.DataFrame(raw_data, columns=['subject_id', 'first_name', 'last_name'])

raw_data = {
    'subject_id': ['4', '5', '6', '7', '8'],
    'first_name' : ['Billy', 'Brain', 'Bran', 'Bryce', 'Betty'],
    'last_name' : ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']
}

df_b = pd.DataFrame(raw_data, columns=['subject_id', 'first_name', 'last_name'])

print(pd.merge(df_a, df_b, on='subject_id', how='outer'))
