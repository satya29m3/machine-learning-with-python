import numpy as np
import matplotlib.pyplot as plt

data  = np.genfromtxt('ex1data1.txt' , delimiter = ',')


xs = np.array(data[:,0])
ys = np.array(data[:,1])
print xs[0:4],ys[0:4]

