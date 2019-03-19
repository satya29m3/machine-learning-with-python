import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from statistics import mean
from sklearn import preprocessing , svm
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression


data = np.genfromtxt('data/ex1data1.txt', delimiter=',')

X= np.array(data[:,0])
y = np.array(data[:,1])

#for i in range(len(X)):
#	print X[i] , y[i]



def best_fit_slope(X,y):
	m = (((mean(X)*mean(y)) - (mean(X*y)) ) / 

		((mean(X)**2)-(mean(X**2)))
		)

	b =mean(y) - (m*mean(X))

	return m,b

def squared_error(ys_orig, ys_line):
	return sum((ys_line-ys_orig)**2)

def coeff_of_determination(y_orig,y_line):
	y_mean_line = [ mean(y_orig) for i in y_orig]
	squared_er_reg = squared_error(y_orig, y_line)
	squared_er_mean= squared_error(y_orig, y_mean_line)
	return 1 - (squared_er_reg/squared_er_mean)

m,b =best_fit_slope(X,y)


regression_line = [(m*x)+b for x in X]

r_squared = coeff_of_determination(y,regression_line)


print r_squared , m ,b

predict_x=[7,3.5]
predict_y=[ (m*i)+b for i in predict_x]

print predict_y

plt.scatter(X,y,marker="o",s=10, color = 'r')
plt.scatter(predict_x,predict_y,marker="o", color = 'blue',)

plt.plot(X,regression_line, color= 'b')
plt.xlabel('population of the city in 10,000s')
plt.ylabel('price of the houses in $10,000s')
plt.show()

