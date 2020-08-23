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

dir = "gs://"
nepochs = 10
requests.packages.urllib3.disable_warnings()
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

def upload(filename):
    print("upload "+filename+" to "+dir)
    subprocess.check_call(
        ['gsutil', 'cp', filename, dir+"/"+filename])

def data_load(ticker, period, price):
    tick = yf.Ticker(ticker)

    # get stock info
    tick.info

    # get historical market data
    hist = tick.history(period=period)
    print(hist.head())
    print(list(hist.columns))
    series = hist[price].values
    initial = series[0]
    print("initial is {}".format(initial))
    series = series / initial
    #print(series[:10])
    time = np.array(list(hist.index))
    #ds = tf.data.Dataset.from_tensor_slices(series)
    #print(tf.compat.v1.data.get_output_shapes(ds))
    #print(next(ds.as_numpy_iterator()))
    return (time,series,initial)

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def split_train_val(time,series,thd=0.8):
    N = len(series)
    split_time = int(N*thd)
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]
    return (time_train,x_train,time_valid,x_valid,split_time)


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(100).prefetch(1)
    forecast = model.predict(ds)
    return forecast

def model_build():
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                          strides=1, padding="causal",
                          activation="relu",
                          input_shape=[None, 1]),
      tf.keras.layers.LSTM(60, return_sequences=True),
      tf.keras.layers.LSTM(60, return_sequences=True),
      tf.keras.layers.Dense(30, activation="relu"),
      tf.keras.layers.Dense(10, activation="relu"),
      tf.keras.layers.Dense(1)
      #tf.keras.layers.Lambda(lambda x: x * 400) # MSFT, AAPL
      #tf.keras.layers.Lambda(lambda x: x * 500)  # NFLX
    ])


    #optimizer = tf.keras.optimizers.SGD(lr=5e-5, momentum=0.5)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-8 * 10**lr)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    print(model.summary())
    return model

def accuracy(x_valid,rnn_forecast):
    a = []
    b = []
    n = min(len(x_valid),len(rnn_forecast))
    for i in range(1,n):
        a.append(x_valid[i]-x_valid[i-1])
        b.append(rnn_forecast[i]-rnn_forecast[i-1])
    a = np.array([1 if x > 0 else -1 for x in a])
    b = np.array([1 if x > 0 else -1 for x in b])
    c = a*b
    acc = sum ([ 1 if x > 0 else 0 for x in c])
    acc = float(acc) / n

    return acc

def evaluate(model,time,series,split_time,window_size,filename):
    rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
    print(rnn_forecast.shape)
    rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]
    print(rnn_forecast.shape)
    plt.figure(figsize=(6,4))
    time_valid = time[split_time:]
    x_valid = series[split_time:]
    plot_series(time_valid,x_valid)
    plot_series(time_valid,rnn_forecast)
    #plt.show()
    plt.savefig(filename+"-prediction.png")
    upload(filename+"-prediction.png")
    mae = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
    print("mae is {}".format(mae))
    print("mean is {}".format(sum(x_valid)/len(x_valid)))
    acc = accuracy(x_valid,rnn_forecast)
    print("accuracy is {}".format(acc))
    return (rnn_forecast,mae,acc)
    #return (rnn_forecast,mae)

def one_prediction(model,initial,current,window_size):
    current = tf.expand_dims(current, axis=-1)
    d = tf.data.Dataset.from_tensor_slices(current).window(window_size, drop_remainder=True)
    print(tf.compat.v1.data.get_output_shapes(d))
    d = d.flat_map(lambda w: w.batch(window_size))
    print(tf.compat.v1.data.get_output_shapes(d))
    d = d.batch(1)
    print(tf.compat.v1.data.get_output_shapes(d))
    #for w in d:
        #print(w)
    future = model.predict(d)
    future = future * initial
    #future_forecast = model_forecast(model, current[..., np.newaxis], window_size)
    #future_forecast = future_forecast * initial
    print("current price is ")
    print(current[-1] * initial)
    #print(current[-1])
    print("future prediction price is ")
    print(future.shape)
    print(future[:,-1,0])
    return future[:,-1,0]
#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------

def loss_plot(history,filename):
    loss=history.history['loss']
    epochs=range(len(loss)) # Get number of epochs
    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.plot(epochs, loss, 'r')
    plt.title('Training loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss"])

    #plt.figure()
    plt.savefig(filename+"-loss.png")
    upload(filename+"-loss.png")


    end=len(loss)
    start=int(end*2.0/3.0)
    zoomed_loss = loss[start:end]
    zoomed_epochs = range(start,end)


    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.plot(zoomed_epochs, zoomed_loss, 'r')
    plt.title('Zoomed Training loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss"])
    #plt.figure()

    #plt.show()
    plt.savefig(filename+"-zoomed-loss.png")
    upload(filename+"-zoomed-loss.png")


def step_prediction(model,initial,series,window_size):
    prediction_list = []
    last = len(series) - window_size
    current = series[last:]
    for i in range(5):
        future1step = one_prediction(model,initial,current,window_size)
        future1step = future1step[0]
        print("1 step prediction is {}".format(future1step))
        prediction_list.append(future1step)
        a = list(current[1:])
        a.append(future1step/initial)
        #a.append(future1step)
        current = np.array(a)
    return prediction_list

def one_loop(ticker,period,price):
    filename = "-".join([ticker,period,price])
    (time, series, initial) = data_load(ticker, period, price)
    thd=0.8
    (time_train, x_train, time_valid, x_valid, split_time) = split_train_val(time, series, thd)

    tf.keras.backend.clear_session()
    #tf.random.set_seed(51)
    #np.random.seed(51)
    shuffle_buffer_size = 1000
    window_size = 60
    batch_size = 100
    train_set = windowed_dataset(x_train, window_size=60, batch_size=100, shuffle_buffer=shuffle_buffer_size)
    print(tf.compat.v1.data.get_output_shapes(train_set))
    model = model_build()
    #logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    #file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    #file_writer.set_as_default()
    class MyMetricCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            self.hpt = hypertune.HyperTune()
        def on_epoch_end(self, epoch, logs=None):
            #tf.summary.scalar('lr1', logs['mae'], epoch)
            self.hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag='lr1',
                metric_value=logs['mae'],
                global_step=epoch
            )

    history = model.fit(train_set,epochs=nepochs,callbacks=[MyMetricCallback()])
    loss_plot(history,filename)

    (rnn_forecast, mae, acc) = evaluate(model,time, series, split_time, window_size,filename)
    print("mae is {}".format(mae))
    u = sum(x_valid)/len(x_valid)
    print("mean is {}".format(sum(x_valid)/len(x_valid)))
    #mae = mae / u
    prediction_list = step_prediction(model, initial, series, window_size)
    #result = {"mae": str(mae), "5-step-prediction": str(future1step[0])}
    #result = {"mae": str(mae), "5-step-prediction": str(prediction_list) }
    result = [str(mae), str(acc)]
    heading = ["mae", "accuracy"]
    for i,v in enumerate(prediction_list):
        result.append(str(v))
        heading.append(str(i+1)+"-step")
    print(heading)
    print(result)
    with open(filename+"-number.csv", "w") as outfile:
        outfile.write(",".join(heading))
        outfile.write("\n")
        outfile.write(",".join(result))
        outfile.write("\n")
    upload(filename+"-number.csv")

def parser(ticker,period,price):
    filename = "-".join([ticker, period, price])
    with open(filename + "-number.csv", "r") as csvfile:
        lines = csvfile.readlines()
        if (len(lines) >= 2):
            row = lines[1].split(",")
            numbers = [float(x) for x in row]
    return numbers


def trading(list,period):
    result = []
    #period='3y'
    # Facebook (FB), Amazon (AMZN), Apple (AAPL), Netflix (NFLX); and Alphabet (GOOG)
    #list = ["FB","AMZN","AAPL","NFLX","GOOG"]
    #ticker='AAPL'
    #price='Open'
    for ticker in list:
        price='Open'
        numbers = parser(ticker,period,price)
        mae = numbers[0]
        acc = numbers[1]
        a = sum(numbers[2:])/(len(numbers)-2)
        d = numbers[-1] - numbers[2]
        item = {}
        item["ticker"]=ticker
        item["mae"]=mae
        item["acc"]=acc
        item["Open"]=a
        item["delta"]=d
        result.append(item)
        #price='High'
        #numbers = parser(ticker,period,price)
        #mae = numbers[0]
        #item['mae'] = max(item['mae'],mae)
        #item['High'] = numbers[1]
        #price='Close'
        #numbers = parser(ticker,period,price)
        #mae = numbers[0]
        #item['mae'] = max(item['mae'],mae)
        #item['Close'] = numbers[1]
        #item['delta'] = (item['Close'] - item['Open'])/item['Open']
        #result.append(item)
    print(result)
    df = pd.DataFrame(result).sort_values('delta', ascending=False)
    print(df)
    return df

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
        default=6,
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
    list = ["FB","AMZN"]
    #ticker='AAPL'
    #price='Open'
    for ticker in list:
        one_loop(ticker,period,'Open')
        #one_loop(ticker,period,'High')
        #one_loop(ticker,period,'Close')
    trading(list, period)

if __name__ == '__main__':
    args = get_args()
    nepochs = args.num_epochs
    lr = args.learning_rate
    dir = args.job_dir
    # Run the training job
    train_and_evaluate(args)