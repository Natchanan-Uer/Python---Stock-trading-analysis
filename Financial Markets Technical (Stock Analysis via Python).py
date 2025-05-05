import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import norm 
from pandas.plotting import scatter_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf
import yfinance as yf 
from datetime import date
from dateutil.relativedelta import relativedelta

ticker = input("Enter the stock trading of your choice's symbol. E.g., AAPL for Apple, TSLA for Tesla, AMZN for Amazon: ")
stockdata = yf.download(ticker, start=date.today()-relativedelta(years=5), end=date.today())
stockdata.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
stockdata['Logreturn'] = np.log(stockdata['Close'].shift(-1)) - np.log(stockdata['Close'])

#Moving Average of the stock
stockdata['MA50'] = stockdata['Close'].rolling(50).mean()
stockdata['MA100'] = stockdata['Close'].rolling(100).mean()
stockdata['MA200'] = stockdata['Close'].rolling(200).mean()

plt.figure(1)
plt.plot(stockdata['Close'], label = "Close")
plt.plot(stockdata['MA50'], label = "MA50")
plt.plot(stockdata['MA100'], label = "MA100")
plt.plot(stockdata['MA200'], label = "MA200")
plt.title("Stock Price Comparison")
plt.legend()

#12-days and 16-days Exponential Moving Average 
stockdata['EMA12'] = stockdata['Close'].ewm(com=12).mean()
stockdata['EMA26'] = stockdata['Close'].ewm(com=26).mean()

plt.figure(2)
plt.plot(stockdata['EMA12'], label = "EMA12")
plt.plot(stockdata['EMA26'], label = "EMA26")
plt.title("Stock Price EMAs")
plt.legend()

#Moving Average Convergence Divergence Indicator (MACD)
#MACD = EMA12 - EMA26)
stockdata['MACD'] = stockdata['EMA12'] - stockdata['EMA26']
stockdata['Signal'] = stockdata['MACD'].ewm(com=9).mean()
stockdata['MACDHist'] = stockdata['MACD'] - stockdata['Signal']
plt.figure(3)
plt.plot(stockdata['MACD'], label = "MACD")
plt.plot(stockdata['Signal'], label = 'Signal Line')
plt.plot(stockdata['MACDHist'], label= 'MACD Histogram')
plt.axhline(y=0, color='black', linestyle = "-")
plt.title("Moving Average Convergence Divergence Indicator")
plt.legend()

#Oscillator and Momentum Indicator (RS and RSI) 
stockdata['Diff'] = stockdata['Close'].shift(-1) - stockdata['Close']

stockdata['Loss'] = stockdata['Diff'].apply(lambda x: -x if x <= 0 else 0)
stockdata['Gains'] = stockdata['Diff'].apply(lambda x: x if x > 0 else 0)

stockdata['AvgGain'] = stockdata.iloc[:,16].ewm(com=14).mean()
stockdata['AvgLoss'] = stockdata.iloc[:,15].ewm(com=14).mean()
stockdata['RS'] = stockdata['AvgGain']/stockdata['AvgLoss']
stockdata['RSI'] = 100 - (100/(1+stockdata['RS']))
plt.figure(4)
plt.plot(stockdata.iloc[-365:-1,20], label= 'RSI')
plt.axhline(y=30, color='blue', linestyle = '--', label='Oversold')
plt.axhline(y=70, color='red', linestyle = '--', label='Overbrought')
plt.title("RSI Oscillator plot")
plt.legend()
plt.show()

#2nd stock for correlation (SP500 and ticker2 as predictors, ticker as a response)
SP500s = '^GSPC' 
SP500 = yf.download(SP500s, start=date.today()-relativedelta(years=5), end=date.today())
SP500.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
ticker2 = input("Enter the stock you want as a predictor. E.g., AAPL for Apple, TSLA for Tesla, AMZN for Amazon: ")
stockdata2 = yf.download(ticker2, start=date.today()-relativedelta(years=5), end=date.today())
stockdata2.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
if len(stockdata2) == len(stockdata):
    None
else:
    print("The stock has misaligned dates due to recent IPO")
    exit()

stockcorr = pd.DataFrame()
stockcorr[ticker] = stockdata['Close'].shift(-1) - stockdata['Close'] 
stockcorr[ticker2] = stockdata2['Close'].shift(-1) - stockdata2['Close']
stockcorr['SP500'] = SP500['Close'].shift(-1) - SP500['Close']
stockcorr = stockcorr.dropna()

x = stockcorr[[ticker2, 'SP500']]
x = sm.add_constant(x)
y = stockcorr[ticker]
stockmodel = sm.OLS(y,x).fit()
fp_value = float(stockmodel.summary().tables[0].data[3][3])
alpha = 0.05
dfn = 2 #dof numerator
dfd = len(stockcorr) #dof denominator
critical_f = stats.f.ppf(1 - alpha, dfn, dfd)
print("\n"
    "--- Test for Significance of the regression model, F-test at 95% Confidence Interval ---\n"
        "                           H0 = Beta1 = Beta2 =...= Betak = 0\n"
        "                           H1: At least one Beta is not zero\n",)
if fp_value <= critical_f:
    print("fp_value <= critical_f \n"
    "The model contains useful predictors\n"
    "------------------------------------End of F-test---------------------------------------"
    "\n")
else: 
    print("fp_value > critical_f \n"
    "The model is insignificance\n"
    "------------------------------------End of F-test----------------------------------------"
    "\n")


