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

#symbol= 'BTC-USD'

# start = dt.datetime(2016,1,1)
# end = dt.datetime(2019,1,1)
# df = web.DataReader(symbol, 'yahoo', start, end)
# df.to_csv('test.csv')
# print (df.info())
# df = pd.read_csv('test.csv',parse_dates=True,index_col=0 )
# df = df.round(4)
# df.to_csv('test.csv')
df = pd.read_csv('test.csv')
df = df.set_index('Date')
print (df.info())
style.use('ggplot')

# df.plot(kind= 'box',subplots=True, layout= (1,6),sharex=False,sharey=False)
# df.hist()
# scatter_matrix(df)
# plt.show()
#df.rename(column= {})
# percentage change for open vs close, and high vs low

df ['OpenVsClose_change'] = (df['Close']-df['Open'])/ df['Open'] * 100
#
df ['HighVsLow_change'] = (df['High']-df['Low'])/ df['Low'] * 100
# #
# # just change the data set
new_df = pd.DataFrame(df)
new_df = new_df[['OpenVsClose_change','HighVsLow_change','Volume','Close']]
#plt.scatter(df['OpenVsClose_change'],df['Close'])
#df.plot(kind= 'box',subplots=True, layout= (1,8),sharex=False,sharey=False)
print(new_df.head(100))
#  # will forcast the Closing price
#
forecast_col = 'Close'
# # get the data set length percentage's 0.1 will be in the forecasted
forecast_out = int(math.ceil(0.1* len(new_df)))# 1 % of the data

# preparaing for the empty labels for the incoming forcast

new_df['label']= new_df[forecast_col].shift(-forecast_out)
print (new_df)


new_df.dropna(inplace=True)
#
print (new_df.tail())

# # get x value and y value of as rest of the data column and label column
# x = np.array(new_df.drop(['Close'],1))
# y = np.array(new_df['Close'])
x = new_df.iloc[:,: -1]
y = new_df.iloc[:,-1]
# print ("X is --->\n",x)
# print ("\ny is --->\n",y)

# x = np.array(new_df.drop(['label'],1)) # all columns, other than label column
# y = np.array(new_df['label'])# only label column
#
# # get testing set and training set
#
# #split the dataset with a random seed
# # training size is the 90% of the data set
#
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
# print ("X train is ",x_train)
# print ("y train is ",y_train)
# print ("X test is ",x_test)
# print ("y test is ",y_test)
#
# # train with the logistiReression model algorithm
# # LogRModel= LogisticRegression()
# #
# # LogRModel.fit(x_train,y_train)
# #
# # #Predition the model with test data
# # #predition =LogRModel.predict(x_test)
# #
# # # print("predition is: ")
# # # print(predition)
# #
# # # get accurency value for the predited test data
# # accurency = LogRModel.score(x_test,y_test)
# # print ("accurency score is: ")
# #
# # print ("Logistic Model", accurency)
#
# Linear regression model
linear_regression = LinearRegression()
linear_regression.fit(x_train,y_train)
#coef= linear_regression.coef_
# print("coef array is : ", coef)
# print ("\nintercept is: -->",linear_regression.intercept_)
#predit = linear_regression.predict(x)

# #predit =linear_regression.predict([[0.212114,1.136645,101774924],[0.299099,1.467785,73894412],[0.353047,2.024179,130291591]])
# print ("predition is : ", predit)
# print ("accurency is : ", linear_regression.score(x,y))
# plt.plot(df['Volume'],df['Close'])
# plt.show()

# # not to come out the negative value in the accurency score
# LinearRegression(copy_X= True, fit_intercept=True,n_jobs=1, normalize=False)
# accuracy = linear_regression.score(x_test,y_test)
# # #
# print ('\nLinear_regression accuracy is :', accuracy)


# # # New predit
#
# X = x[:-forecast_out]
#
# # # old one
#
# X_lately = x[-forecast_out:]
# #print(X_lately)
# print(new_df.tail(30))
# # #predit the stock price for the bitcoin for next 0.1% of the day which is 4 day for here
# Forecast_set = linear_regression.predict(X_lately)
# print (Forecast_set)

# # just visualization
