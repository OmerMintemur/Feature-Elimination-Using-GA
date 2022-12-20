from sklearn import linear_model
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import numpy as np


df_train = pd.read_csv('TrainTransformed.csv')
df_test = pd.read_csv('TestTransformed.csv')
X_train, y_train = df_train.loc[:, df_train.columns != 'SalePrice'], df_train.loc[:, df_train.columns == 'SalePrice']
X_test, y_test = df_test.loc[:, df_test.columns != 'SalePrice'], df_test.loc[:, df_test.columns == 'SalePrice']
test=15
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)


reg_RF = RandomForestRegressor(n_estimators = 100, random_state = 0)
reg_RF.fit(X_train, y_train)
print("Random Forest Regressor Train Performance ", mean_squared_error(np.log(y_train), np.log(reg_RF.predict(X_train)),squared=True))
print("Random Forest Regressor Test Performance ", mean_squared_error(np.log(y_test), np.log(reg_RF.predict(X_test)),squared=True))
# print(int(reg_RF.predict((X_test.iloc[test]).values.reshape(1,-1))))
# print(y_test.iloc[test])


reg_ABR = AdaBoostRegressor(n_estimators = 100, random_state = 0)
reg_ABR.fit(X_train, y_train)
print("Adaboost Regressor Train Performance ", mean_squared_error(np.log(y_train), np.log(reg_ABR.predict(X_train)), squared=True))
print("Adaboost Regressor Test Performance", mean_squared_error(np.log(y_test), np.log(reg_ABR.predict(X_test)), squared=True))
# print(int(reg_RF.predict((X_test.iloc[test]).values.reshape(1,-1))))
# print(y_test.iloc[test])

Lasso = linear_model.Lasso()
Lasso.fit(X_train, y_train)
print("Lasso Train Performance ", mean_squared_error(np.log(y_train), np.log(reg_ABR.predict(X_train)), squared=True))
print("Lasso Test Performance", mean_squared_error(np.log(y_test), np.log(reg_ABR.predict(X_test)), squared=True))
# print(int(reg_RF.predict((X_test.iloc[test]).values.reshape(1,-1))))
# print(y_test.iloc[test])



