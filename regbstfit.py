from statistics import mean
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('ggplot')

data = np.genfromtxt('ex1data1.txt', delimiter=',')

xs= np.array(data[:,0])
ys= np.array(data[:,1])


#xs =np.array( [1,2,3,4,5], dtype = np.float64)
#ys = np.array([5,4,6,5,6], dtype =np.float64) 
"""
def create_dataset(hm,variance, step=2, correlation=False):
	val = 1
	ys = []
	for i in range(hm):
		y= val + random.randrange(-variance, variance)
		ys.append(y)
		if correlation and correlation == 'pos':
			val+=step
		elif correlation and correlation == 'data = np.genfromtxt('ex1data1.txt', delimiter=',')

X= np.array(data[:,0])
y = np.array(data[:,1])neg':
			val-=step

	xs = [i for i in range(len(ys))]
	return np.array(xs,dtype=np.float64) , np.array(ys , dtype= np.float64)"""


def best_fit_slp(xs,ys):
	m= (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)**2) - mean(xs**2)))

	b= mean(ys) - (m*mean(xs))

	return m,b

def squarederror(ys_orig, ys_line):
	return sum((ys_line-ys_orig)**2)

def Coefficient_of_determination(ys_orig ,ys_line):
	y_mean_line = [mean(ys_orig) for y in ys_orig]
	squared_error_regr = squarederror(ys_orig ,ys_line)
	squared_error_y_mean= squarederror(ys_orig ,y_mean_line)
	return 1-(squared_error_regr/squared_error_y_mean)


#xs,ys= create_dataset(40 ,40,2,correlation = 'pos')

m, b= best_fit_slp(xs,ys)

#regression_line = [(m*x)+b for x in xs]

regression_line = []
for x in xs:
	regression_line.append((m*x)+b)

r_squared = Coefficient_of_determination(ys,regression_line)
print(r_squared)



print(m,b)

predict_x = 7
predict_y= (m*predict_x)+b
plt.scatter(xs,ys,color='r',label = 'data', marker ='x')
plt.scatter(predict_x ,predict_y ,color='g')
plt.plot(xs,regression_line,label= 'regression line', color= 'b')
plt.legend(loc=4)
plt.show()

