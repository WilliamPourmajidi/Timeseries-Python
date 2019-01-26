import scipy.stats as st
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa import stattools as stt
from statsmodels.tsa.vector_ar.var_model import forecast
import statsmodels.api as smapi
import datetime
import matplotlib.pyplot as plt
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
from fbprophet import Prophet
import csv
import os

plt.style.use(os.path.join(os.getcwd(), 'mystyle.mplstyle'))
plt.rcParams['axes.edgecolor'] = 'w'
import warnings  # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go
# from pandas.io import data, wb
from pandas_datareader import data, wb
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 15, 6
# plt.style.use('fivethirtyeight')


# Settings for Panda dataframe displays

pd.options.display.width = 1200
pd.options.display.max_colwidth = 100
pd.options.display.max_columns = 100

# Import Data Set
df = pd.read_csv('numenta_ec2_cpu_utilization_5f5533.csv'
                 # ,
                 # parse_dates=['timestamp'],
                 # index_col='timestamp',
                 # delimiter=','
                 )

trimmed_df = df[['timestamp', 'value']]  # Trimming columns to timestamp and value
trimmed_df.columns = ['ds', 'y']  # renaming coloumn as Prophet is picky on names!
print(trimmed_df)

model = Prophet()  # Creating the model
model.fit(trimmed_df)  # Fitting the data frame to the model

# future = model.make_future_dataframe(periods=3)
# print(future.tail())

future = model.make_future_dataframe(periods=10000, freq='5Min')
print(future)
forecast = model.predict(future)

# print("##############Before\n",trimmed_df)
trimmed_df['yhat'] = forecast['yhat']
trimmed_df['yhat_lower'] = forecast['yhat_lower']
trimmed_df['yhat_upper'] = forecast['yhat_upper']
# #
# #

print("##############After\n", trimmed_df)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

stdv = np.std(trimmed_df['y'])
print(stdv)
upper_anomalyborder = trimmed_df['yhat_upper'] + int(1.5 * stdv)
lower_anomalyborder = trimmed_df['yhat_lower'] - int(1.5 * stdv)
trimmed_df['upper_anomalyborder'] = upper_anomalyborder
trimmed_df['lower_anomalyborder'] = lower_anomalyborder
print(trimmed_df)

### Anomaly Detection

anomalies = pd.DataFrame(index=trimmed_df.index)
anomalies['Anomalies'] = np.nan

print(anomalies)

# # updating the anomalies data frame with anomaly values
for i, row in trimmed_df.iterrows():  # i: dataframe index; row: each row in series format
    if ((row['y']) > (row['upper_anomalyborder']) or (row['y']) < (row['lower_anomalyborder'])  ):
        # print("Warning: Anomaly Detected")
        anomalies.loc[i]['Anomalies'] = row['y']

print(anomalies)
# #


#
#
# # fig1 = model.plot(forecast)
# # fig1.show()
#
#
plt.figure(figsize=(18, 5))
plt.title("Prophet(ARIMA)-based Data Analysis", fontsize=20)
plt.plot(trimmed_df['y'], label='Actual Data')
plt.plot(trimmed_df['yhat'], label='Predicted Data')
plt.plot(trimmed_df['yhat_upper'], "r-.", label="Predicted Upper Bound")
plt.plot(trimmed_df['yhat_lower'], "r-.", label="Predicted Lower Bound")
plt.plot(trimmed_df['upper_anomalyborder'], label="Upper Anomaly Border")
plt.plot(trimmed_df['lower_anomalyborder'], label="Lower Anomaly Border")
plt.plot(anomalies['Anomalies'], "ro", markersize=10, label="Anomalies")
plt.legend(loc='best')
plt.grid(True)
plt.show()


# Pure ARIMA


# create an empty dataframe with the same index and columns of testing dataset
# ## Enable for extra plot
# plt.figure(figsize=(18, 5))
# plt.title("Simple Average Forecast",fontsize=20)
# plt.plot(training_set.index, training_set['av'], label='Training Dataset')
# plt.plot(testing_set.index, testing_set['av'], label='Testing Dataset')
# plt.plot(y_hat_avg['avg_forecast'], label='Simple Average Forecast')
#
# plt.legend(loc='best')
# plt.grid(True)
# plt.show()

# updating the anomalies data frame with anomaly values
# for i, row in y_hat_avg.iterrows():  # i: dataframe index; row: each row in series format
#     # print(row['av'],i)
#     if ((row['av']) > (row['upper_bound'])):
#         anomalies_average.loc[i]['av'] = row['av']
#
# plt.figure(figsize=(18, 5))

# plt.plot(anomalies_naive, "ro", markersize=2, label="Anomalies")


# plt.plot(testing_set.index, testing_set['av'], label='Testing Dataset')
# plt.plot(y_hat_naive.index, y_hat_naive['naive'], label='Naive Forecast')
