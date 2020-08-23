import yfinance as yf
import urllib3
import requests
import ssl
import tensorflow as tf
import numpy as np
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
print(tf.__version__)

requests.packages.urllib3.disable_warnings()
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# Facebook (FB), Amazon (AMZN), Apple (AAPL), Netflix (NFLX); and Alphabet (GOOG)
tick = yf.Ticker("AAPL")

# get stock info
tick.info

# get historical market data
hist = tick.history(period="3y")
print(hist.head())
print(list(hist.columns))
series = hist['Open'].values
initial = series[0]
print("initial is {}".format(initial))
series = series / initial
N = len(series)
print(series[:10])
time = np.array(list(hist.index))
#ds = tf.data.Dataset.from_tensor_slices(series)
#print(tf.compat.v1.data.get_output_shapes(ds))
#print(next(ds.as_numpy_iterator()))

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

split_time = int(N*0.8)
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

#window_size = 30
#batch_size = 32
shuffle_buffer_size = 1000

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

tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
window_size = 60
batch_size = 100
train_set = windowed_dataset(x_train, window_size=60, batch_size=100, shuffle_buffer=shuffle_buffer_size)
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
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set,epochs=1000)

rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]
plt.figure(figsize=(6,4))
plot_series(time_valid,x_valid)
plot_series(time_valid,rnn_forecast)
plt.show()
mae = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
print("mae is {}".format(mae))
print("mean is {}".format(sum(x_valid)/len(x_valid)))


#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
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

plt.figure()


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

plt.show()

