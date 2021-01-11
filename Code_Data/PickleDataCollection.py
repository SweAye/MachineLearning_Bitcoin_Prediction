from sklearn import svm
import quandl
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import datetime as dt
import pandas as pd
import time
import pickle
import pandas_datareader.data as web
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import math
import datetime
import numpy as np
from sklearn.naive_bayes import GaussianNB
from pandas.plotting import scatter_matrix
from sklearn.metrics import accuracy_score
import pickle


df = pd.read_csv('test.csv')
#print (df.info())
df = df.set_index('Date')
#print (df.head())
style.use('ggplot')

df ['OpenVsClose_change'] = (df['Close']-df['Open'])/ df['Open'] * 100
df ['HighVsLow_change'] = (df['High']-df['Low'])/ df['Low'] * 100
new_df = pd.DataFrame(df)
new_df = new_df[['Close','HighVsLow_change','OpenVsClose_change','Volume']]
# recording the real Closing price before testing for the forecasting the Price
original_DataFrame =pd.DataFrame(columns=['Date','Price'])
original_DataFrame =pd.DataFrame(new_df['Close'].values , columns=['Price'])
original_DataFrame = original_DataFrame[1068:1098] # This Before the 30 day of the end day
forecast_col = 'Close'
forecast_out = int(math.ceil(0.027* len(new_df)))
new_df['label']= new_df[forecast_col].shift(-forecast_out)
X = np.array(new_df.drop(['label'],1)) # all columns, other than label column
X = X[:-forecast_out]
X_lately = X[-forecast_out:]
new_df.dropna(inplace=True)
y = np.array(new_df['label'])# only label column
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
linear_regression = LinearRegression(n_jobs = -1)
linear_regression.fit(x_train,y_train)
with open ('linearregressionFitted.pickle', 'wb') as f:
    pickle.dump(linear_regression,f)
LinearRegression(copy_X= True, fit_intercept=True,n_jobs=1, normalize=False)
accuracy = linear_regression.score(x_test,y_test)
print ('\nLinear_regression accuracy is :', accuracy)

