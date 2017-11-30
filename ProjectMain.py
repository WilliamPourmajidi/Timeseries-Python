import scipy.stats as st
import seaborn as sns
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
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
#from pandas.io import data, wb
from pandas_datareader import data, wb
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
from fbprophet import Prophet
#plt.style.use('fivethirtyeight')



# def despine(axs):
#     # to be able to handle subplot grids
#     # it assumes the input is a list of
#     # axes instances, if it is not a list,
#     # it puts it in one
#     if type(axs) != type([]):
#         axs = [axs]
#     for ax in axs:
#         ax.yaxis.set_ticks_position('left')
#         ax.xaxis.set_ticks_position('bottom')
#         ax.spines['bottom'].set_position(('outward', 10))
#         ax.spines['left'].set_position(('outward', 10))



####

def is_stationary(df, maxlag=14, autolag=None, regression='ct'):
    """Run the Augmented Dickey-Fuller test from statsmodels and print output.
    """
    outpt = stt.adfuller(df,maxlag=maxlag, autolag=autolag,
                            regression=regression)
    print('adf\t\t {0:.3f}'.format(outpt[0]))
    print('p\t\t {0:.3g}'.format(outpt[1]))
    print('crit. val.\t 1%: {0:.3f}, \
5%: {1:.3f}, 10%: {2:.3f}'.format(outpt[4]["1%"],
                                     outpt[4]["5%"], outpt[4]["10%"]))
    print('stationary?\t {0}'.format(['true', 'false']\
                                   [outpt[0]>outpt[4]['5%']]))
    return outpt



############### Import Data Set #############

# dateparse = lambda d: pd.datetime.strptime(d, '%Y-%m-%d')
# temp = pd.read_csv('mean-daily-temperature-fisher-river2.csv',
#                    parse_dates=['Date'],
#                    index_col='Date',
#                    date_parser=dateparse,
#                    )


df = pd.read_csv('stationary.csv',
                 parse_dates=['ds'],
                 index_col='ds',
                 delimiter=','
                 )


print("\n#### Information about the imported Time Series and its fields####\n" )
df.info()

print("\n##### Top 5 rows ####\n", df.head())

df = df.iloc[:, 0]  # Purely integer-location based indexing for selection by position.

print("\n##### Indexing on Time! ####\n")

print(df.index)

print("\n#####Statsicical details of the time series ####\n", df.describe())

print("\n#####Checking the stationrity of the time series based on Augmented Dickey-Fuller ####\n" )
print(is_stationary(df))

moving_avg = pd.rolling_mean(df,12)
decomposition = seasonal_decompose(df,freq=50)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.plot(df)
plt.plot(moving_avg, label="Moving Average", color='red')
plt.plot(trend, label='Trend', color='yellow')
# plt.plot(seasonal,label='Seasonality',color='orange')

plt.legend(['Raw', 'Moving Average','Trend'])

plt.show()


### Prophet Forcasting
df2 = pd.read_csv('Stationary.csv', delimiter=',')
df2['y'] = np.log(df2['y'])



my_model = Prophet(interval_width=0.95)

my_model.fit(df2)

#future_dates = my_model.make_future_dataframe(periods=3, freq='MS')
future_dates = my_model.make_future_dataframe(periods=30)

future_dates.tail()

forecast = my_model.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

my_model.plot(forecast,uncertainty=True)
plt.show()

my_model.plot_components(forecast)
plt.show()






plotly.tools.set_credentials_file(username='salman.pourmajidi', api_key='1LdgMkdGWb4TodAi8Y9S')
sns.set(style="white", color_codes=True)

pr = go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat'],
    name="yhat",
    line=dict(color='#003366'),
    opacity=0.8)

lower = go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat_lower'],
    name="yhat_lower",
    line=dict(color='#333331'),
    opacity=0.8)

upper = go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat_upper'],
    name="yhat_upper",
    line=dict(color='#733331'),
    opacity=0.8)

data = [pr, lower, upper]

layout = dict(
    title='Time Series with Rangeslider',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(),
        type='date'
    )
)

fig = dict(data=data, layout=layout)
py.plot(fig, filename="Time Series with Rangeslider")








#########################################################


#
# df = pd.read_csv('DataFile3.csv', delimiter=',')
# # df['y'] = np.log(df['y'])
#
# df.head()
# print(df)
#
# is_stationary(df)

# m = Prophet()
# m.fit(df)
#
# future = m.make_future_dataframe(periods=60)
# future.tail()
#
# forecast = m.predict(future)
# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
#
# m.plot(forecast, xlabel="Timeee")
# m.plot_components(forecast)
# # plot.show()
#
#
# # Credential for Plotly
# plotly.tools.set_credentials_file(username='salman.pourmajidi', api_key='1LdgMkdGWb4TodAi8Y9S')
# sns.set(style="white", color_codes=True)
#
# Dallas = go.Scatter(
#     x=forecast['ds'],
#     y=forecast['yhat'],
#     name="yhat",
#     line=dict(color='#003366'),
#     opacity=0.8)
#
# Dallas2 = go.Scatter(
#     x=forecast['ds'],
#     y=forecast['yhat_lower'],
#     name="yhat_lower",
#     line=dict(color='#333331'),
#     opacity=0.8)
#
# Dallas3 = go.Scatter(
#     x=forecast['ds'],
#     y=forecast['yhat_upper'],
#     name="yhat_upper",
#     line=dict(color='#733331'),
#     opacity=0.8)
#
# data = [Dallas, Dallas2, Dallas3]
#
# layout = dict(
#     title='Time Series with Rangeslider',
#     xaxis=dict(
#         rangeselector=dict(
#             buttons=list([
#                 dict(count=1,
#                      label='1m',
#                      step='month',
#                      stepmode='backward'),
#                 dict(count=6,
#                      label='6m',
#                      step='month',
#                      stepmode='backward'),
#                 dict(step='all')
#             ])
#         ),
#         rangeslider=dict(),
#         type='date'
#     )
# )
#
# fig = dict(data=data, layout=layout)
# py.plot(fig, filename="Time Series with Rangeslider")
#
#
#
