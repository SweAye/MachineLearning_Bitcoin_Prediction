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
from sklearn import preprocessing

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

# there are 1098 tuples, - 30 is 1067
original_DataFrame = original_DataFrame[1068:1098] # This Before the 30 day of the end day
plt.plot (original_DataFrame['Price'])
# # print (newDf)
# # plt.plot(newDf['Price'])
# plt.title("Original Date vs Price graph")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
forecast_col = 'Close'
forecast_out = int(math.ceil(0.027* len(new_df)))# 30 days of the data out of 1098 days , accurency with 75% to 83% swinging, +-8% change
new_df['label']= new_df[forecast_col].shift(-forecast_out)

X = np.array(new_df.drop(['label'],1)) # all columns, other than label column
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]
new_df.dropna(inplace=True)
y = np.array(new_df['label'])# only label column

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

pickled = open('linearregressionFitted.pickle','rb')
linear_regression = pickle.load(pickled)

#not to come out the negative value in the accurency score
LinearRegression(copy_X= True, fit_intercept=True,n_jobs=1, normalize=False)
accuracy = linear_regression.score(x_test,y_test)
print ('\nLinear_regression accuracy is :', accuracy)

svm_modle= svm.SVR()
svm_modle.fit(x_train,y_train)

accuracy_SVR = svm_modle.score(x_test,y_test)
print ("svR_accurency:", accuracy_SVR)
#predit the stock price for the bitcoin for next 0.1% of the day which is 4 day for here
Forecast_set = linear_regression.predict(X_lately)

print ("Forecast_set is :", Forecast_set)
new_df['Forecast'] = np.nan

last_date = new_df.iloc[-1].name

# print ("last date is: " , last_date)
last_date = time.mktime(datetime.datetime.strptime(last_date,"%Y-%m-%d").timetuple())
one_day = 86400
next_unix = last_date + 86400

label_arry = np.array(new_df['label'])

next_date_array= []
for i in Forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)# might be this one wrong
    next_date = str(next_date)
    #print ("String next date is:", next_date)
    next_date= str.split(next_date," ")
    #print (("Splited string next date is:", next_date[0]))
    next_date = next_date[0]
    next_date_array.append(next_date) # just to get an arrray for later use
    next_unix +=one_day
    new_df.loc[next_date] = [np.nan for _ in range(len(new_df.columns)-1)] +[i]
print ("next_date array is: ", next_date_array)
newDf = pd.DataFrame(columns=['Date','Price'])
newDf = pd.DataFrame({'Date': next_date_array[:],'Price': Forecast_set[:]})
print ("newDf is: ---->", newDf)

Forecast_DataFrame= pd.DataFrame(Forecast_set)
print ("Forecast DataFrame is ", Forecast_DataFrame)
Forecast_DataFrame.to_csv('Forecast.csv')
new_df['Forecast'].plot()

plt.title("January 1st, 2016 To January 1st, 2019 bitcoin Stock price and 30 day forecast")
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

