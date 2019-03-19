import pandas as pd
import quandl, math ,datetime
import  numpy as np
from sklearn import preprocessing , svm
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.pyplot import style

style.use('ggplot')






df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100.0



df= df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]



forecast_col='Adj. Close'
df.fillna(-99999, inplace =True )
forecast_out=int(math.ceil(0.01*len(df)))
print(forecast_out)
df['label']=df[forecast_col].shift(-forecast_out)




X= np.array(df.drop(['label'],1))
X= preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y=np.array(df['label'])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf= LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)



"""for k in ['linear','poly','rbf','sigmoid']:
	clf = svm.SVR(kernel=k)
	clf.fit(X_train,y_train)
	accuracy = clf.score(X_test,y_test)
	print(k,accuracy)
"""
forecast_set= clf.predict(X_lately)

print(forecast_set,accuracy,forecast_out)

df['Forecast']= np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day=86400
next_unix = last_unix + one_day



for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += 86400
	df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]
print X_lately[0:3]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('date')
plt.ylabel('price')
plt.show()























































