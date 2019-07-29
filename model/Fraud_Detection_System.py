import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

## data load
data = pd.read_csv('./data/creditcard.csv')
print(data.head())
print(data.columns) # check columns name
## check data freq
print(pd.value_counts(data['Class']))

pd.value_counts(data['Class'].plot.bar())
plt.tilte('Fraud class histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()