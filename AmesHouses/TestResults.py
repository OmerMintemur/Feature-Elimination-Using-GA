import pandas as pd
import random
import numpy as np
from itertools import compress
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

df_train = pd.read_csv('TrainTransformed.csv')
df_test = pd.read_csv('TestTransformed.csv')
X_train, y_train = df_train.loc[:, df_train.columns != 'SalePrice'], df_train.loc[:, df_train.columns == 'SalePrice']
X_test, y_test = df_test.loc[:, df_test.columns != 'SalePrice'], df_test.loc[:, df_test.columns == 'SalePrice']
FULL_FEATURES = list(X_train.columns)
print(FULL_FEATURES)

X_train_new_features = X_train[['Street', 'LandSlope', 'RoofMatl', 'Heating', '3SsnPorch', 'MiscVal', 'YrSold']]
X_test_new_features =  X_test[['Street', 'LandSlope', 'RoofMatl', 'Heating', '3SsnPorch', 'MiscVal', 'YrSold']]


X_train_new_features = X_train_new_features.astype('float32')
X_test_new_features = X_test_new_features.astype('float32')

min_max_scaler = preprocessing.MinMaxScaler()
X_train_Normal = min_max_scaler.fit_transform(X_train_new_features)
X_test_Normal = min_max_scaler.transform(X_test_new_features)

reg_RF = lgb.LGBMRegressor()
reg_RF.fit(X_train_Normal, y_train)

#print(np.sqrt(mean_squared_error(np.log(y_test), np.log(reg_RF.predict(X_test_Normal)))))

son = reg_RF.predict(X_test_Normal).reshape(len(X_test_Normal),1)


print(np.log(y_test) - np.log(son))

