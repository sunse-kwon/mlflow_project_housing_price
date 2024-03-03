import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import random
random.seed(42)

# read data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# x, y split for train set
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']

# split train, val, split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# split numerical and categorical variables
numerical = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical = X_train.select_dtypes(exclude=['int64', 'float64']).columns

# impute missing values for numerical variables
imputer = KNNImputer()
X_train[numerical] = imputer.fit_transform(X_train[numerical])
X_val[numerical] = imputer.transform(X_val[numerical])
test[numerical] = imputer.transform(test[numerical])

# impute missing value for categorical with mode
for column in categorical:
    X_train[column].fillna(X_train[column].mode()[0], inplace=True)
    X_val[column].fillna(X_val[column].mode()[0], inplace=True)
    test[column].fillna(test[column].mode()[0], inplace=True)

OneHot = OneHotEncoder(drop='first', handle_unknown='ignore')

X_train = OneHot.fit_transform(X_train)
X_val = OneHot.transform(X_val)
test = OneHot.transform(test)
