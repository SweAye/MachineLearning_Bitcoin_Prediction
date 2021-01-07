import matplotlib.pyplot as plt
import requests
import pandas as pd
from pandas import DataFrame
import urllib.parse

"""
To change the requre start data and end datae go to this link
https://www.quandl.com/data/BITFINEX/VETBTC-VET-BTC-Exchange-Rate

The is API key 
https://www.quandl.com/api/v3/datasets/BITFINEX/VETBTC.csv?api_key=FE2iNsurxQjRHbvScnLk
"""
# r = requests.get('https://www.quandl.com/api/v3/datasets/BITFINEX/VETBTC.csv?api_key=FE2iNsurxQjRHbvScnLk')
# print(r)
# #sending jason file into the dat frame
# df = pd.DataFrame(r)
# df.to_csv('quandle.csv')
# print (df.info())
# print(df.head())
# drop the null values
# df.dropna(inplace=True)
#
# #split a cloumn to two column
#
# #since "bpi" is one columns with values and key dictionary type
#
# # sending values of the bpi as price
# Price = df['bpi'].values
# print (Price)
#
# # sending key (Dates) of the bpi as Date
# Date= df['bpi'].keys()
# print (Date)
#
# # making a new data frame with columns labels
# newDf = pd.DataFrame(columns=['Date','Close'])
# newDf['Date']= Date # fill the column with Date
# newDf['Close']= Price# fill the column with Prices

#newDf.to_csv('coindesk.csv')
# df = pd.read_csv('quandle.csv',parse_dates=True,index_col=0 )
# df = df.round(4)
# print(df.head())
main_api = 'https://www.quandl.com/api/v3/datasets/BITSTAMP/USD.json?api_key=FE2iNsurxQjRHbvScnLk'
#r = requests.get('https://www.quandl.com/api/v3/datasets/BITSTAMP/USD.json?api_key=FE2iNsurxQjRHbvScnLk').json()

json_data = requests.get(main_api).json()
print (json_data)
json_status = json_data['dataset']
print ("Json status: ", json_status)
formatted_address = json_data['dataset']['id']
print ("\nformatted_address:", formatted_address)
#df = pd.DataFrame(r, columns=['bpi'])

#data = r.dataset.dataset_code
#print (data)

# df = pd.DataFrame(r, columns= ['column_names'])
# df.to_csv('quandle.csv')
# print (df.info())

#sending jason file into the dat frame
# df = pd.DataFrame(r)
# df.to_csv('quandle.csv')
# print (df.info())
# print(df.head())
#r = requests.get('https://www.quandl.com/api/v3/datasets/BITSTAMP/USD.csv?api_key=FE2iNsurxQjRHbvScnLk')
#print(r)
#https://www.quandl.com/api/v3/datasets/BITSTAMP/USD.csv?api_key=FE2iNsurxQjRHbvScnLk