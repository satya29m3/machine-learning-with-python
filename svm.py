import pandas as pd 
import os
import  numpy as np
from sklearn import preprocessing , svm , neighbors
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
from math import sqrt
import warnings
from collections import Counter
import random

path = os.getcwd() + '/data/breast-cancer-wisconsin.data.txt'
df = pd.read_csv(path, header= None , names =['sample_code_num','clump_thickness',
											'uni_cell_size','uni_cell_shape',
											'marginal_adhesion',
											'single_epithilial_cell_size','bare_nuclei',
											'bland_chromatin','normal_nucleoli','mitoses',
											'class'] )
df.replace('?',-99999,inplace = True)
df.drop(['sample_code_num'],1,inplace=True)


x=np.array(df.drop(['class'],1))
y= np.array(df['class'])

X_train ,X_test,y_train,y_test = train_test_split(x,y ,test_size=0.2)

clf = svm.SVC()

clf.fit(X_train,y_train)
acc = clf.score(X_test,y_test)
print(acc  )

example_measure = np.array([4,2,1,1,1,2,3,2,1])
example_measure = example_measure.reshape(1,-1)

prediction = clf.predict(example_measure)
print prediction
































