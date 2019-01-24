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
plt.style.use(os.path.join(os.getcwd(), 'mystyle.mplstyle') )
plt.rcParams['axes.edgecolor'] = 'w'
import warnings  # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go
#from pandas.io import data, wb
from pandas_datareader import data, wb
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
#plt.style.use('fivethirtyeight')


# Settings for Panda dataframe displays

pd.options.display.width = 1200
pd.options.display.max_colwidth = 100
pd.options.display.max_columns = 100



# Import Data Set
df = pd.read_csv('numenta_ec2_cpu_utilization_5f5533.csv'
                 #,
                 #parse_dates=['timestamp'],
                 #index_col='timestamp',
                 #delimiter=','
                 )

trimmed_df = df[['timestamp', 'value']] # Trimming columns to timestamp and value
trimmed_df.columns = ['ds', 'y']  # renaming coloumn as Prophet is picky on names!
print(trimmed_df)



model = Prophet()  # Creating the model
model.fit(trimmed_df)  # Fitting the data frame to the model

# future = model.make_future_dataframe(periods=3)
# print(future.tail())


future = model.make_future_dataframe(periods=10, freq='H')




forecast = model.predict(future)


print("##############Before\n",trimmed_df)
trimmed_df['yhat'] = forecast['yhat']
print("##############After\n",trimmed_df)


# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])


fig1 = model.plot(forecast)
# fig1.show()



#
# print("\n#### Information about the imported Time Series and its fields####\n" )
# df.info()
#
# print("\n##### Top 5 rows ####\n", df.head())
# print(df)

#
# #
# # df = df.iloc[:, 0]  # Purely integer-location based indexing for selection by position.
#
# print("\n##### Indexing on Time! ####\n")
# print(df.index)
# print("\n#####Statsicical details of the time series ####\n", df.describe())
#
# #print("\n#####Checking the stationrity of the time series based on Augmented Dickey-Fuller ####\n" )
# #print(is_stationary(df))
#
#
