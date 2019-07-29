import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df=pd.DataFrame([['yellow', 'M', '23', 'a'],
                    ['red', 'L', '26', 'b'],
                    ['blue', 'XL', '20', 'c']])
df.columns = ['color', 'size', 'price', 'type']
print(df)
# print(type(df))

#데이터셋 필요한 부분 숫자변경
x=df[['color', 'size', 'price', 'type']].values # 데이터 프레임에서 numpy.narray로 변회
# print(x)
# print(type(x))

shop_le=LabelEncoder() #string을 int라벨로
x[:,0] = shop_le.fit_transform(x[:,0])
x[:,1] = shop_le.fit_transform(x[:,1])
x[:,2] = x[:,2].astype(dtype=float)
x[:,3] = shop_le.fit_transform(x[:,3])

print("라벨인코터 변환값\n", x)

#원본데이터값
ohe = OneHotEncoder(categorical_features=[0]) #인뎃스 0데이터 원핫코딩
ohe = ohe.fit_transform(x).toarray()
print("라벨 후 원핫인코딩값\n", ohe)