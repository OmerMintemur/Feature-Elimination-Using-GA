import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from MultiColumnsEncoder import MultiColumnLabelEncoder as MLE
desired_width=640
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 80)

def intersect(a, b):
    if len(b) < len(a):  # iff b is shorter than a
        a, b = b, a      # swap the lists.
    b = b[:]  # To prevent modifying the lists
    return [b.pop(b.index(i)) for i in a if i in b]


df_train = pd.read_csv('..\\train.csv')
df_test = pd.read_csv('..\\test.csv')
df_test_price = pd.read_csv('..\\sample_submission.csv')
df_test = pd.merge(df_test, df_test_price, on = 'Id', how='left')

df_train = df_train.dropna(axis=1)
df_test = df_test.dropna(axis=1)
test_columns_list = df_test.columns.to_list()
train_columns_list = df_train.columns.to_list()

final_list = intersect(train_columns_list, test_columns_list)

df_train = df_train[final_list]
df_test = df_test[final_list]

df_train_cat = df_train.select_dtypes(include=['object'])
df_test_cat = df_test.select_dtypes(include=['object'])

df_train_encoded = MLE(columns=df_train_cat.columns.to_list()).fit_transform(df_train)
df_test_encoded = MLE(columns=df_test_cat.columns.to_list()).fit_transform(df_test)
df_test_encoded["SalePrice"] = df_test_encoded["SalePrice"].astype(int)

df_train_encoded.drop('Id', axis=1, inplace=True)
df_test_encoded.drop('Id', axis=1, inplace=True)

print(df_train_encoded)
print("----------------")
print(df_test_encoded)

df_train_encoded.to_csv("TrainTransformed.csv",index=False)
df_test_encoded.to_csv("TestTransformed.csv",index=False)



