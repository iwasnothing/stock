from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yfinance as yf
import requests
import ssl
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import subprocess
from datetime import datetime
import hypertune
from sklearn.preprocessing import MinMaxScaler

def avg(list):
    return sum(list)/len(list)

def data_load(ticker):
    tick = yf.Ticker(ticker)
    period = '4y'
    # get stock info
    #print(tick.info)

    # get historical market data
    hist = tick.history(period=period)
    print(hist.head())
    series = hist['Open'].values
    scaler = MinMaxScaler()
    series = scaler.fit_transform(series.reshape(-1,1))
    print(series.shape)
    series = series.reshape(-1)
    print(series.shape)
    N = len(series)
    thd = 0.8
    split_time = int(thd * N)
    X = []
    Y = []
    for i in range(N-1):
        if i >= 60:
            list = [60,30,10,5]
            row = []
            feature = []
            for w in list:
                row.append(avg(series[i-w:i+1]))
            for a in range(len(row)):
                if a > 0:
                    feature.append(row[a] - row[a-1])
            label = 1 if series[i+1] > series[i] else 0
            X.append(np.array(feature))
            Y.append(label)
    x_train = np.array(X[:split_time])
    y_train = np.array(Y[:split_time])
    x_valid = np.array(X[split_time:])
    y_valid = np.array(Y[split_time:])
    print(x_train.shape)
    return (series,x_train,y_train,x_valid,y_valid,scaler)

def model_build():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(3)),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    #optimizer = tf.keras.optimizers.SGD(lr=5e-5, momentum=0.5)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-8 * 10**lr)
    model.compile(loss = 'binary_crossentropy',
                  optimizer=optimizer,
                  metrics = ['accuracy'])
    return model

def evaluate(model,x_valid,y_valid):
    forecast = model.predict(x_valid)
    #print(forecast)
    #print(forecast.shape)
    x = []
    a = 0
    n = min(len(y_valid),len(forecast))
    for i in range(n):
        #print(x_valid[i],forecast[i])
        if y_valid[i] == 1 and forecast[i,0] > 0.5:
            a = a + 1
        if y_valid[i] == 0 and forecast[i,0] < 0.5:
            a = a + 1
        acc = float(a) / n
    print("accuracy is {}".format(acc))
    return (forecast,acc)

def get_args():
    """Argument parser.
    Returns:
        Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        default="output/",
        help='GCS location to write checkpoints and export models')
    parser.add_argument(
        '--train-file',
        type=str,
        help='Dataset file local or GCS')
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Split between training and test, default=0.2')
    parser.add_argument(
        '--num-epochs',
        type=float,
        default=50,
        help='number of times to go through the data, default=500')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3,
        help='learning rate for gradient descent, default=.001')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    args, _ = parser.parse_known_args()
    return args


def train_and_evaluate(args):
    period='4y'
    # Facebook (FB), Amazon (AMZN), Apple (AAPL), Netflix (NFLX); and Alphabet (GOOG)
    # list = ["FB","AMZN","AAPL","NFLX","GOOG"]
    if not os.path.isfile('list.txt'):
        print("download list.txt")
        subprocess.check_call(
            ['gsutil', 'cp', 'gs://iwasnothing-cloudml-job-dir/list.txt', '.'] )
    with open('list.txt','r') as fp:
        list = fp.read().splitlines()
        #print(list)
    list = ["FB"]
    # list = ["ALGN"]
    #ticker='AAPL'
    #price='Open'
    print(list)
    for ticker in list:
        print(ticker)
        (series, x_train, y_train, x_valid, y_valid, scaler) = data_load(ticker)
        tf.keras.backend.clear_session()
        model = model_build()
        class MyMetricCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                self.hpt = hypertune.HyperTune()

            def on_epoch_end(self, epoch, logs=None):
                # tf.summary.scalar('lr1', logs['mae'], epoch)
                self.hpt.report_hyperparameter_tuning_metric(
                    hyperparameter_metric_tag='lr1',
                    metric_value=logs['accuracy'],
                    global_step=epoch
                )

        history = model.fit(x_train,y_train, epochs=nepochs, callbacks=[MyMetricCallback()])
        print(model.summary())
        (forecast, acc) = evaluate(model,x_valid,y_valid)

if __name__ == '__main__':
    args = get_args()
    nepochs = args.num_epochs
    lr = args.learning_rate
    dir = args.job_dir
    # Run the training job
    train_and_evaluate(args)

