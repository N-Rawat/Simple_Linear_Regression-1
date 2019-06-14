# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 02:03:30 2019

@author: ASHOK
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#provide data
X=np.array([1,5,10,15,20,25]).reshape((-1,1))
#.reshape()on x becoz this array is reqd to be 2d, or to have one col and as many as rows necessary
y=np.array([1,4,16,12,8,24]) 
print('X value: ' , X)
print('y value: ' , y)

#Create a model and fit it
model = LinearRegression()
#var model is instance of LinearRegression
#v.fit() calc optimal val of the weights b0 and b1 
model.fit(X,y)

#result
r_sq = model.score(X,y)
print('coefficient of determination:',r_sq)
#R2 is coefficient of determination 
# the attribute of mobel .intercept_ represent the coefficient of b0 and .coef_ represent b1
print('intercept:', model.intercept_)   #scalar value
print('slope:', model.coef_)    #array
#b0 illustrate the predicition of model a value when x is 0 and b1 predicted response roses by  value at an inc of 1 in x

#both x and y can be 2d array
new_model = LinearRegression().fit(X , y.reshape((-1,1)))
print('new intercept:' , new_model.intercept_)  #array of 1 d with 1 value b0
print('new slope:', new_model.coef_)    #array of 2d with 1 value b1

#predict response
y_pred = model.predict(X)
#when .predict(), u pass the regressor as thr arg and get the corresponding predicted response
print('Predicted response:', y_pred, sep='\n')

print('old y predicted value: ' , y_pred)

print('Shape of X : ' , X.shape)
print('Shape of old y predict in 1d: ' , y_pred.shape)


#our manual way
y_pred1 = model.intercept_ + model.coef_ * X
print('predicted response:', y_pred1, sep='\n')

print('Shape of X : ' , X.shape)
print('Shape of y predict in 2d: ' , y_pred1.shape)

#in this case u multiply each ele of x with model.coef_ and add it to model.intercept_
# the o/p here only differ by array size 
#if we reduce x to one d, these two approaches will yield the same res. 
#replace x with x.reshape(-1,1),x.flatten() and x.ravel() when multiplying with model.coef_

X_new = np.arange(6).reshape((-1,1))
print('new X array(input): ' , X_new)
# arrage() from numpy to generate an array with ele from 0 to 5

y_newpred = model.predict(X_new)
print('new y predicted value: ' , y_newpred)

print('Shape of X : ' , X_new.shape)
print('Shape of y predict in 1d: ' , y_newpred.shape)


plt.scatter(X, y, color = 'red')
plt.plot(X, model.predict(X), color = 'blue')
plt.title('X VS y')
plt.xlabel('X')
plt.ylabel('y')
plt.show()


