# Time series - Python
import numpy as np
# %matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
import pandas as pd
from fbprophet import Prophet

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

df = pd.read_csv('AirPassengers.csv')

print(df.head(5))

print(df.dtypes)
df['Month'] = pd.DatetimeIndex(df['Month'])
df.dtypes

print(df.dtypes)

df = df.rename(columns={'Month': 'ds',
                        'AirPassengers': 'y'})

print(df.head(5))

ax = df.set_index('ds').plot(figsize=(12, 8))
ax.set_ylabel('Monthly Number of Airline Passengers')
ax.set_xlabel('Date')

plt.show()

my_model = Prophet(interval_width=0.95)

my_model.fit(df)

future_dates = my_model.make_future_dataframe(periods=196, freq='MS')
future_dates.tail()

forecast = my_model.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

my_model.plot(forecast, uncertainty=True)
plt.show()

my_model.plot_components(forecast)
plt.show()
