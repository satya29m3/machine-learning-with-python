import pandas as pd 
import os
import quandl, math ,datetime
import  numpy as np
from sklearn import preprocessing , svm
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.pyplot import style


style.use('ggplot')

path = os.getcwd() + '/data/ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

#data2 = (data2 - data2.mean())/ data2.std()

cols = data2.shape[1]
X = data2.iloc[:,0:2]
y = data2.iloc[:,cols-1]

clf = LinearRegression(n_jobs=-1)
clf.fit(X,y)

f = clf.predict([[1650 , 3]])



print f

