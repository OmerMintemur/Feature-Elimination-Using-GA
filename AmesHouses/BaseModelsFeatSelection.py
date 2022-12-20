
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression

import numpy as np
import warnings
warnings.filterwarnings("ignore")


# https://www.yourdatateacher.com/2021/10/11/feature-selection-with-random-forest/
def return_new_FeaturesRFE(X_tr, y_tr):
    features = list(X_tr.columns.values)
    rf = RandomForestRegressor(random_state=0)

    rf.fit(X_tr, y_tr)

    f_i = list(zip(features, rf.feature_importances_))
    f_i.sort(key=lambda x: x[1])
    plt.barh([x[0] for x in f_i], [x[1] for x in f_i])

    plt.show()

    rfe = RFE(rf, n_features_to_select=12)

    rfe.fit(X_train, y_train)

    selected_features = np.array(features)[rfe.get_support()]
    print(selected_features)
    return selected_features

def return_new_FeaturesSelectKBest(X_tr, y_tr):

    fs = (SelectKBest(f_regression, k=12)).fit(X_tr, y_tr)
    print(type(fs))
    features =X_tr.columns[fs.get_support()]
    print(features)

    return features

def models(X_tr, y_tr, X_te, y_te):
    reg_RF = RandomForestRegressor(n_estimators=100, random_state=0)
    reg_RF.fit(X_tr, y_tr)
    print("Random Forest Regressor Train Performance ",
          mean_squared_error(np.log(y_tr), np.log(reg_RF.predict(X_tr))))
    print("Random Forest Regressor Test Performance ",
          mean_squared_error(np.log(y_te), np.log(reg_RF.predict(X_te))))


df_train = pd.read_csv('TrainTransformed.csv')
df_test = pd.read_csv('TestTransformed.csv')
X_train, y_train = df_train.loc[:, df_train.columns != 'SalePrice'], df_train.loc[:, df_train.columns == 'SalePrice']
X_test, y_test = df_test.loc[:, df_test.columns != 'SalePrice'], df_test.loc[:, df_test.columns == 'SalePrice']
test=15


X_train_Normal = X_train.astype('float32')
X_test_Normal = X_test.astype('float32')

X_train_FNormal=X_train_Normal
X_test_FNormal=X_test_Normal

print("Before Feature Selection ")
min_max_scaler = preprocessing.MinMaxScaler()
X_train_Normal = min_max_scaler.fit_transform(X_train_Normal)
X_test_Normal = min_max_scaler.transform(X_test_Normal)
models(X_train_Normal, y_train, X_test_Normal, y_test)

columns = return_new_FeaturesRFE(X_train, y_train)
X_train_FNormal = X_train_FNormal[columns]
X_test_FNormal = X_test_FNormal[columns]
print(X_train_FNormal.columns)

min_max_scaler = preprocessing.MinMaxScaler()
X_train_FNormal = min_max_scaler.fit_transform(X_train_FNormal)
X_test_FNormal = min_max_scaler.transform(X_test_FNormal)
print("After Feature Selection ")
models(X_train_FNormal, y_train, X_test_FNormal, y_test)




# Random Forest Regressor Train Performance  0.025823662996025752
# Random Forest Regressor Test Performance  0.07690290727784801


