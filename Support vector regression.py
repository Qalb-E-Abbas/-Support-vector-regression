# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 14:54:21 2020

@author: Qalbe
"""

# Importing the libraries
import numpy as py
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Poly_dataSet.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values



# we will always use standardscaler in order to fix the line over data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

sc_y = StandardScaler()
y = sc_y.fit_transform(py.reshape(y,(10,1)))



# Fitting Linear Regression to the dataset
from sklearn.svm import SVR
svr = SVR(kernel ='rbf')
svr.fit(X, y)



# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, svr.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#Prdict by SV Regression, but it will give us value in point
svr.predict(py.reshape(6.5,(1,1)))

# Predicting a new result with SV Regression
predict = sc_y.inverse_transform(svr.predict(sc_X.transform(py.array([[6.5]]))))

