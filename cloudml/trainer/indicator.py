# -*- coding: utf-8 -*-
"""indicator.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tjj2cEESibOQ8192eiZsYdJr4Zqub4Cl
"""

#!pip install yfinance
#!pip install get-all-tickers
#!pip install ta

from sklearn.metrics import mean_absolute_error
import pandas as pd
from google.cloud import pubsub_v1
import json
import os
import subprocess
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf, pandas as pd, shutil, os, time, glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from get_all_tickers import get_tickers as gt
from ta import add_all_ta_features
from ta.utils import dropna
import requests
import ssl

requests.packages.urllib3.disable_warnings()
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

def one_loop(ticker,price="Close",period="4y"):
    #ticker = "AAPL"
    tick = yf.Ticker(ticker)
    print(period)
    hist = tick.history(period=period)
    hist['Timestamp'] = hist.index.values
    if len(hist.index) <= 100:
        return None
    df = add_all_ta_features(hist, open="Open", high="High", low="Low", close="Close", volume="Volume")
    df100 = df.iloc[100:]
    cols = df.columns.values
    features = cols[8:]

    price = df100[price].values

    df100 = df100[features].dropna(axis=1)

    features = df100.columns.values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df100.values)

    pca=PCA(n_components=3)
    newData=pca.fit_transform(X)

    print(newData[0])

    initial = price[0]
    ratio = price / initial
    Y = ratio [1:]

    X = newData[:-1]
    currentX = newData[-1]
    if np.isnan(X).any() or np.isnan(Y).any():
            return None
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)
    # The mean squared error
    print('Mean squared error: %f' % mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute error: %f' % mae")
    future = regr.predict([currentX])
    print(Y[-1], future[0], (future[0] - Y[-1])/Y[-1])
    print(Y[-1]*initial, future[0]*initial, (future[0] - Y[-1])/Y[-1])

    result = {"ticker": ticker,
              "mae": mae,
              "current": Y[-1]*initial,
              "future": future[0]*initial,
              "delta": (future[0] - Y[-1])/Y[-1]
              }


    #plt.plot(y_test)
    #plt.plot(y_pred)
    #plt.show()
    return result



def train_and_evaluate():
    if not os.path.isfile('list.txt'):
        print("download list.txt")
        subprocess.check_call(
            ['gsutil', 'cp', 'gs://iwasnothing-cloudml-job-dir/list.txt', '.'] )
    with open('list.txt','r') as fp:
        list = fp.read().splitlines()
        print(list)
    df = pd.DataFrame()
    count = 0
    for ticker in list:
        print(ticker)
        period = "4y"
        price = "Close"
        result = one_loop(ticker,price,period)
        if result != None:
            count = count + 1
            df = df.append(result, ignore_index=True)
            print(df)
    print(count)
    print(df.head())
    #df = df[df.delta > 0]
    df = df.sort_values("mae", ascending=True)
    ticker = df.iloc[0]["ticker"]
    inc2 = df.iloc[0]["delta"]
    project_id = "iwasnothing-self-learning"
    topic_id = "submit_order_topic"
    publisher = pubsub_v1.PublisherClient()
    # The `topic_path` method creates a fully qualified identifier
    # in the form `projects/{project_id}/topics/{topic_id}`
    topic_path = publisher.topic_path(project_id, topic_id)
    j = {"ticker": ticker, "spread": inc2}
    data = json.dumps(j)
    print(data)
    # Data must be a bytestring
    data = data.encode("utf-8")
    # When you publish a message, the client returns a future.
    future = publisher.publish(topic_path, data=data)
    print(future.result())
    print("Published messages.")

if __name__ == '__main__':
    train_and_evaluate()
