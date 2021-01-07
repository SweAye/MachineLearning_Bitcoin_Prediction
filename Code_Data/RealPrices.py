"""
This program will analyse the short term predition with the last 3 day of price
"""

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
from sklearn import svm
import quandl
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import datetime as dt
import pandas as pd
import pandas_datareader.data as web
from matplotlib import style
import matplotlib.pyplot as plt
import math
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.metrics import accuracy_score

symbol= 'BTC-USD'

def got_start_Date():
    return datetime.datetime.now() + datetime.timedelta(-30)



def got_end_Date():
     return dt.datetime.today()


#


# symbol= 'BTC-USD'
# start =datetime.datetime(2018,12,4)
#
# end = datetime.datetime(2019,1,2)
#
# df = web.DataReader(symbol, 'yahoo', start, end)
# df.to_csv('RealPrice.csv')
# print (df.info())

df_Forcast = pd.read_csv('Forecast.csv',parse_dates=True,index_col=0 )
df_Forcast = df_Forcast.round(4)
df_Forcast.to_csv('Forecast.csv')

df_Forcast = pd.read_csv('Forecast.csv')

#print (df_Forcast)
df_Forcast.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
df_Forcast.drop(["a"], axis=1, inplace=True)
#print ("df Forcast is",df_Forcast)

df = pd.read_csv('RealPrice.csv',parse_dates=True,index_col=0 )
df = df.round(4)
df.to_csv('RealPrice.csv')
df = pd.read_csv('RealPrice.csv')
df = df.set_index('Date')
#print (df.info())
style.use('ggplot')







#----------------------

# df.plot(kind= 'box',subplots=True, layout= (1,6),sharex=False,sharey=False)
# df.hist()
# scatter_matrix(df)
# plt.show()
#df.rename(column= {})
# percentage change for open vs close, and high vs low

df ['OpenVsClose_change'] = (df['Close']-df['Open'])/ df['Open'] * 100
#
df ['HighVsLow_change'] = (df['High']-df['Low'])/ df['Low'] * 100
#
# # just change the data set
new_df = pd.DataFrame(df)
new_df = new_df[['Close','HighVsLow_change','OpenVsClose_change','Volume']]
#print("new_df is: --->", new_df)

# just the Closing column

close_Price = np.array(new_df['Close'])
forecast_price = np.array((df_Forcast.iloc[:,0]))



print("forecast Price are : ", len(forecast_price))

print ("closePrice are: ",len(close_Price))

#--------------

# # red dashes, blue squares and green triangles
plt.plot(close_Price,'r--')
plt.show()


#--------------
#plt.plot(close_Price)
#plt.plot (forecast_price)

#plt.show()
#
#
#
# # recording the real Closing price before testing for the forecasting the Price
# original_DataFrame =pd.DataFrame(columns=['Date','Price'])
#
#
# original_DataFrame =pd.DataFrame(new_df['Close'].values , columns=['Price'])
#
# # there are 1098 tuples, - 30 is 1067
# original_DataFrame = original_DataFrame[1068:1098] # This Before the 30 day of the end day
# #print ("original_DataFrame is ----->", original_DataFrame)
#
#
# #original_DataFrame = pd.DataFrame({'Date': next_date_array[:],'Price': Forecast_set[:]})
# plt.plot (original_DataFrame['Price'])
# # # print (newDf)
# # # plt.plot(newDf['Price'])
# # plt.title("Original Date vs Price graph")
# plt.xlabel("Date")
# plt.ylabel("Price")
# #plt.show()
#
#
#
#
# #  # will forcast the Closing price
# #
# forecast_col = 'Close'
# # # # get the data set length percentage's 0.1 will be in the forecasted
# forecast_out = int(math.ceil(0.027* len(new_df)))# 30 days of the data out of 1098 days , accurency with 75% to 83% swinging, +-8% change
# #print ("Forcast_out is : ", forecast_out)
# #
# # # # preparaing for the empty labels for the incoming forcast
# #
# new_df['label']= new_df[forecast_col].shift(-forecast_out)


# #print (new_df['label'])
#
# # # # get x value and y value of as rest of the data column and label column
# #
# X = np.array(new_df.drop(['label'],1)) # all columns, other than label column
# X = X[:-forecast_out]
# X_lately = X[-forecast_out:]
# print("Xlately is:", X_lately)
# new_df.dropna(inplace=True)
# y = np.array(new_df['label'])# only label column
#
#
# #print (len(X), len(y))
# # # # get testing set and training set
# # #
# # # #split the dataset with a random seed
# # # # training size is the 90% of the data set
# # #
# x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
#
#
# # # Linear regression model
# # linear_regression = LinearRegression()
# # linear_regression.fit(x_train,y_train)
# # with open ('linearregressionFitted.pickle', 'wb') as f:
# #      pickle.dump(linear_regression,f)
#
#
#
# pickled = open('linearregressionFitted.pickle','rb')
# linear_regression = pickle.load(pickled)
#
# # # not to come out the negative value in the accurency score
# LinearRegression(copy_X= True, fit_intercept=True,n_jobs=1, normalize=False)
# accuracy = linear_regression.score(x_test,y_test)
#
# print ('\nLinear_regression accuracy is :', accuracy)
#
#
#
#  #predit the stock price for the bitcoin for next 0.1% of the day which is 4 day for here
# Forecast_set = linear_regression.predict(X_lately)
#
# print ("Forecast_set is :", Forecast_set)


# new_df['Forecast'] = np.nan
#
# last_date = new_df.iloc[-1].name
#
# # print ("last date is: " , last_date)
# last_date = time.mktime(datetime.datetime.strptime(last_date,"%Y-%m-%d").timetuple())
# # print ("timestamp is: ",last_date)
# #
# #
# #
# one_day = 86400
# next_unix = last_date + 86400
# # print ("next unix is: ", next_unix)
#
# ## just to show the forcast_set with Price values
# label_arry = np.array(new_df['label'])
#
# # for j in
# #     for i in Forecast_set
#
# next_date_array= []
# # # just visualization ( later get inside the
# for i in Forecast_set:
#     next_date = datetime.datetime.fromtimestamp(next_unix)# might be this one wrong
#     next_date = str(next_date)
#     #print ("String next date is:", next_date)
#     next_date= str.split(next_date," ")
#     #print (("Splited string next date is:", next_date[0]))
#     next_date = next_date[0]
#     next_date_array.append(next_date) # just to get an arrray for later use
#
#
#
#
#
#     # this should be in function (change it later)
#     next_unix +=one_day
#     new_df.loc[next_date] = [np.nan for _ in range(len(new_df.columns)-1)] +[i]
#
#
# print ("next_date array is: ", next_date_array)
#
# # make a forcast vs nexdate dataset for Demo
#
# newDf = pd.DataFrame(columns=['Date','Price'])
#
#
# newDf = pd.DataFrame({'Date': next_date_array[:],'Price': Forecast_set[:]})
# #original_Data= pd.DataFrame({'Date': next_date_array[:],'Price': original_DataFrame['Price']})
# print ("newDf is: ---->", newDf)
#
#
# # let's do the same dataframe
#
#
# # plt.plot(newDf['Price'])
# #
# # plt.title("Forcasted Price vs Original Price graph")
# # plt.xlabel("Date")
# # plt.ylabel("Prices")
# # plt.show()
# # print (len(next_date_array))
# # print (len(Forecast_set))
#
# #new_df['Date']= next_date_array
# #new_df['Price']= Forecast_set
# # print (new_df)
#
# #------------------------------------------------------
# new_df['Close'].plot()
# new_df['Forecast'].plot()
#
# plt.title("January 1st, 2016 To January 1st, 2019 bitcoin Stock price and 30 day forecast")
# plt.legend(loc=4)
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.show()







