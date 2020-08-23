from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import numpy as np
import os
import subprocess
from datetime import datetime
import hypertune
from fbprophet import Prophet
import argparse
import yfinance as yf
import requests
import ssl
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
import holidays

requests.packages.urllib3.disable_warnings()
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def get_holidays():
    holiday = pd.DataFrame([])
    for date, name in sorted(holidays.UnitedStates(years=[2017, 2018, 2019, 2020]).items()):
        holiday = holiday.append(pd.DataFrame({'ds': date, 'holiday': "US-Holidays"}, index=[0]), ignore_index=True)
    holiday['ds'] = pd.to_datetime(holiday['ds'], format='%Y-%m-%d', errors='ignore')
    return holiday

def fb_predict(hist,p):
    holiday = get_holidays()
    #print(hist.head())
    df = pd.DataFrame()
    df['y'] = hist['Open'].values
    df['ds'] = hist.index
    #print(df.head())
    #m = Prophet()
    m = Prophet(changepoint_prior_scale = p['changepoint_prior_scale'],
    holidays_prior_scale = p['holidays_prior_scale'],
    n_changepoints = p['n_changepoints'],
    seasonality_mode = p['seasonality_mode'],
    weekly_seasonality=True,
    daily_seasonality = True,
    yearly_seasonality = True,
    holidays=holiday,
    interval_width=0.95)
    m.add_country_holidays(country_name='US')
    m.fit(df)
    return m

def data_load(ticker, period, price):
    tick = yf.Ticker(ticker)

    # get stock info
    #print(tick.info)
    params_grid = {'seasonality_mode': ('multiplicative', 'additive'),
                   'changepoint_prior_scale': [0.1, 0.2, 0.3, 0.4, 0.5],
                   'holidays_prior_scale': [0.1, 0.2, 0.3, 0.4, 0.5],
                   'n_changepoints': [100, 150, 200]}
    grid = ParameterGrid(params_grid)
    model_parameters = pd.DataFrame(columns=['MAE', 'Parameters'])
    # get historical market data
    hist = tick.history(period=period)
    #print(hist.head())
    N = hist.shape[0]
    thd = 0.9
    split_time = int(N*thd)
    print(N,split_time)
    id = 0
    for p in grid:
        m = fb_predict(hist[:split_time],p)
        validation = N - split_time
        print(validation)
        future = m.make_future_dataframe(periods=validation)
        forecast = m.predict(future)
        #print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        rnn_forecast = forecast['yhat']
        rnn_forecast = rnn_forecast[split_time:]
        print(rnn_forecast)

        time = np.array(list(hist.index))
        series = hist['Open']
        time_valid = time[split_time:]
        x_valid = series[split_time:]
        mae = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
        model_parameters = model_parameters.append({'id': id,'MAE': mae, 'Parameters': p}, ignore_index=True)

        if mae < 70:
            plot_series(time_valid,x_valid)
            plot_series(time_valid,rnn_forecast)
            #plt.show()
            plt.savefig("grid-" + str(id) + "-fb-prophet-prediction.png")
        id = id + 1


    model_parameters = model_parameters.sort_values(by='MAE',ascending=True)
    model_parameters.to_csv('model_parameters.csv')
    p = list(grid)
    i = model_parameters.iloc[0]['id']
    print(p[i])
    print(model_parameters)

if __name__ == '__main__':
    data_load('GOOG', '2y', 'Open')