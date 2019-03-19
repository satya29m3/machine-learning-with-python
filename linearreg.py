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
print (df)
	
















