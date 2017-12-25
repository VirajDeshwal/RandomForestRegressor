#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 19:18:07 2017

@author: virajdeshwal
"""

import pandas as pd

file = pd.read_csv('Position_Salaries.csv')

X = file.iloc[:,1:2].values
y=file.iloc[:,2].values


from sklearn.ensemble import RandomForestRegressor

#configure the n_estimators(the collection of trees to increase the accuracy)
model = RandomForestRegressor(n_estimators = 500, random_state = 0)

model.fit(X,y)

#predicting the salary for the experience of 6.5 years.
y_pred = model.predict(6.5)



#Visualizing in the higher dimensions

import numpy as np

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid)),1)
plt.scatter(X,y, color = 'cyan')
plt.plot(X_grid, model.predict(X_grid), color = 'green')
plt.xlabel('Position level')
plt.ylabel('Salary')

plt.show()

print('\n\nthe predicted salary of the employee is :', y_pred)

