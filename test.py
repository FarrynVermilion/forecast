# udah bener cuma kalo bisa benerin format output jadi Y doang bukan y m d
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = pd.read_csv('data.csv', parse_dates=['Date'], index_col='Date')
print(data)

result = adfuller(data['ODA pacific total'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# ARIMA model parameters (p, d, q)
p = 5  # AutoRegressive (AR) order
d = 1  # Differencing order
q = 0  # Moving Average (MA) order

model = ARIMA(data, order=(p, d, q))
model_fit = model.fit()

forecast_steps = 5
forecast = model_fit.forecast(steps=forecast_steps)

latest_date = data.index.values[-1]
s1 = [[latest_date, data['ODA pacific total'].iloc[-1]]]
for x in forecast:
    latest_date=pd.to_datetime(latest_date)+pd.DateOffset(years= 1)
    s1.append([latest_date,x])
df=pd.DataFrame(s1,columns=['Date','ODA Pacific'])
df.set_index('Date', inplace=True)
print(df)

plt.plot(data, label='Original Data')
plt.plot(df , color='red', label='Forecast')
plt.title('Time Series Forecast')
plt.xlabel('Date')
plt.ylabel('ODA pacific total')
plt.legend()
plt.show()