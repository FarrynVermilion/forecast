import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = pd.read_csv('data.csv', parse_dates=['Date'], index_col='Date')
print(data)

# plt.plot(data)
# plt.title('Time Series Data')
# plt.xlabel('Date')
# plt.ylabel('ODA pacific total')
# plt.show()

result = adfuller(data['ODA pacific total'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# ARIMA model parameters (p, d, q)
p = 5  # AutoRegressive (AR) order
d = 1  # Differencing order
q = 0  # Moving Average (MA) order

model = ARIMA(data, order=(p, d, q))
model_fit = model.fit()

forecast_steps = 1
forecast = model_fit.forecast(steps=forecast_steps)
print(forecast)

plt.plot(data, label='Original Data')
plt.plot(range(len(data), len(data) + forecast_steps), forecast, color='red', label='Forecast')
plt.title('Time Series Forecast')
plt.xlabel('Date')
plt.ylabel('ODA pacific total')
plt.legend()
plt.show()
