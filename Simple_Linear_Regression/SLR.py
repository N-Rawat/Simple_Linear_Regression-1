# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 01:52:26 2019

@author: ASHOK
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('home.csv')
X= dataset.iloc[:, :-2].values   #convert dataframe to array
y = dataset.iloc[:, 6].values
print(X.shape)
print(y.shape)
X=pd.DataFrame(X)  #this convert array to dataframe
y=pd.DataFrame(y)
'''#taking care og missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='Nan',strategy='mean',axis=0)
imputer = imputer.fit_transform(X[:,3:6])
#X[:,3:6] = imputer.transform(X[:,3:6])
'''
#splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.25,random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test=sc_y.transform(y_test)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
 
'''
#categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_X = LabelEncoder()
X[:, :1] = le_X.fit_transform(X[:, :1]) #0 because of 1column n : vbecoz of rows
ohe = OneHotEncoder(categorical_features = [0])
X = ohe.fit_transform(X).toarray()
print(X)

# Encoding the Dependent Variable
labelencoder_salary = LabelEncoder()
y = labelencoder_y.fit_transform(y)
'''
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

print(y_pred)
print(y_test)

# Visualising the Training set results
plt.scatter(X_train.reshape(-1,1), y_train.reshape(-1,1), color = 'red')
plt.plot(X_train.index, regressor.predict(X_train), color = 'blue')
plt.title('Price vs Space and View (Training set)')
plt.xlabel('Space')
plt.ylabel('Price')
plt.show()

# Visualising the Test set results
plt.scatter(X_test.reshape(-1,1), y_test.reshape(-1,1), color = 'red')
plt.plot(X_train.index, regressor.predict(X_train), color = 'blue')
plt.title('Price vs Space and View (Test set)')
plt.xlabel('Space')
plt.ylabel('Price')
plt.show()
