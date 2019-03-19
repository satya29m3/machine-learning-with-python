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

def k_nearest_neighbors(data,predict, k=3):
	if len(data) >= k:
		warnings.warn('K is set to value less thantotal voting groups! idiot')
	distances = []
	for group in data:
		for features in data[group]:
			euclidian_distance = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([euclidian_distance,group])

	votes = [i[1] for i in sorted(distances)[:k]]
	#print Counter(votes).most_common(1)
	vote_result = Counter(votes).most_common(1) [0] [0] 
	return vote_result

full_data = df.astype(float).values.tolist()
#print (full_data[0:10])

random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0.0
total = 0.0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1

z= float(correct/total)
print('Accuracy:',z)
print test_set[2]
print test_set[4]

"""
x=np.array(df.drop(['class'],1))
y= np.array(df['class'])

X_train ,X_test,y_train,y_test = train_test_split(x,y ,test_size=0.2)

clf = neighbors.KNeighborsClassifier()

clf.fit(X_train,y_train)
acc = clf.score(X_test,y_test)
print(acc  )

example_measure = np.array([4,2,1,1,1,2,3,2,1])
example_measure = example_measure.reshape(1,-1)

prediction = clf.predict(example_measure)
print prediction
"""






 

#result = k_nearest_neighbors(dataset,new_feature,k=3)
#print result



























