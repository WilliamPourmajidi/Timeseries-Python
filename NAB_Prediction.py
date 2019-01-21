
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

############### Import Data Set #############
df = pd.read_csv('grafana_data_export-2017-10-09-to-2017-10-23.csv',
                 parse_dates=['Time'],
                 index_col='Value',
                 delimiter=';'
                 )

print("\n#### Information about the imported Time Series and its fields####\n" )
df.info()

# print("\n##### Top 5 rows ####\n", df.head())
#
# df = df.iloc[:, 0]  # Purely integer-location based indexing for selection by position.

print("\n##### Indexing on Time! ####\n")
print(df.index)
print("\n#####Statsicical details of the time series ####\n", df.describe())

#print("\n#####Checking the stationrity of the time series based on Augmented Dickey-Fuller ####\n" )
#print(is_stationary(df))


