# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 03:07:30 2019

@author: ASHOK
"""

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
dataset = pd.read_csv('Salaries.csv')
Desig = dataset.iloc[:, 2:3].values   #convert dataframe to array
salary = dataset.iloc[:, 3:9].values
print(Desig.shape)
print(salary.shape)
Desig=pd.DataFrame(Desig)  #this convert array to dataframe
salary=pd.DataFrame(salary)

'''#taking care og missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=0,strategy='mean',axis=0)
imputer = imputer.fit(y[:,4:6])
y[:,4:6] = imputer.transform(y[:,4:6])
'''
#splitting data
from sklearn.model_selection import train_test_split
Desig_train, Desig_test, salary_train, salary_test=train_test_split(Desig, salary, test_size=0.2,random_state=0)
print(Desig_train)
print(Desig_test)
print(salary_train)
print(salary_test)
#categorical data
# Encoding the Independent Variable


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_Desig = LabelEncoder()
Desig[:, 0] = le_Desig.fit_transform(Desig[:, 0]) #0 because of 1column n : vbecoz of rows
ohe = OneHotEncoder(categorical_features = [0])
Desig = ohe.fit_transform(Desig).toarray()
print(Desig)



'''# Encoding the Dependent Variable
labelencoder_salary = LabelEncoder()
salary = labelencoder_salary.fit_transform(salary)
'''

# Fitting Simple Linear Regression to the Training set
x2

# Predicting the Test set results
salary_pred = regressor.predict(Desig_test)
print(salary_pred)
print(salary_test)

'''# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
print(X_test.shape)
print(y_test.shape)
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
'''