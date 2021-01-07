"""
This program will live stream from the yahoo finical anaylis of bitcoin value in US dollar
- use the LinearRegression algorithm to predit the next 10 days of the stock
- other anaylis like stock aggresionness with volume and open and close percentage change
- if time permit , PCA will use the reduction of the dimension inorder to get the more accurency %
- try again with SVM also if time permit
- will use the every 4 repetation of 6 months ( saved data) worth data ( predit, 1 month) for perdition from 2017 January to June,  July to December, 2018 Janunary to Jue,July to Decemeber,
2019 April 1st to April 30  (May 1st to 9th predition) ( live Stream)
- how to show result?
- visualization
- use lineregression graph to compare the predicted data and real data (from the same website, same data as prediction)
- show accurency rate for the predicted data compare with algorithm accurency vs real data predition


quandl API call, with APIkey( go back to the quandle.py and test again)
https://www.quandl.com/api/v3/datasets/BCHAIN/MKPRU.json?api_key=FE2iNsurxQjRHbvScnLk
"""
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

start= dt.datetime(2018,12,6)
end = dt.datetime(2019,6,5)
df = web.DataReader(symbol, 'yahoo', start, end)
df.to_csv('2018_19_12_6.csv')
#print (df)
df = pd.read_csv('2018_19_12_6.csv',parse_dates=True,index_col=0 )
df = df.round(4)
df.to_csv('2018_19_12_6.csv')


