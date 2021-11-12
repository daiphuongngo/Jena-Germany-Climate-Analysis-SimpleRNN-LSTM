# Jena-Germany-Climate-Analysis-SimpleRNN-LSTM

## Language, Machine Learning, Deep Learning models, Tools:

- Python

- SimpleRNN

- LSTM

## Target

Predicting time-series dataset by RNN, including 2 parts: predicting on the time-series dataset with a single variable (1 feature) and predicting on time-series dataset with multiple variables (multiple features).

# Theory Review

## RNN equations

Diagram and equations of vanila RNN for 3 timesteps with inputs $\mathbf{x}_t$, 1 hidden layer $\mathbf{h}_t$, and 1 softmax layer for output $\mathbf{\hat{y}}_t$.

- $\mathbf{{h}}_t = \gamma(W_h\mathbf{h}_{t-1} + W_x\mathbf{x}_{t} + \mathbf{b}_h)$
- $\mathbf{\hat{y}}_t = s(W_y\mathbf{h}_t + \mathbf{b}_y)$

## Weight-sharing in time

What is weight-sharing in RNN and why is it useful?

- When we *unroll* an RNN in time, we have a cascade of FNNs (feed-forward) with exactly *same* set of parameters $(W_h, W_x, W_y, b_h, b_y)$. This is called "weight-sharing in time" and it helps reduce the number of learning parameters, making RNNs very efficient models for training.

## BPTT training

Gradient-based training method for RNN:

- Gradient-based method: backpropagation through time (BPTT) 
  1. Treat the unfolded network as a big feed-forward network. 
  2. The whole input sequence is given to the FFNN. 
  3. The weight updates are computed for each copy in the unfolded network using the usual back-propagation method.
  4. All the updates are then summed (or averaged) and then applied to the RNN (shared) weights.

# Analysis

## Reference

[How to Convert a Time Series to a Supervised Learning Problem in Python (with labels)](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)

## Climate dataset

This [Climate Dataset](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip) has 14 different features: air temperature, air pressure, humidity,... in Jena, Germany, recorded every 10 minutes, from 2003 to 2016. 

I will download the dataset directly from Google storage.

Index	| Features |	Format	| Description
--- | --- | --- | --- 
1	| Date Time	| 01.01.2009 00:10:00	| Date-time reference
2	| p (mbar)	| 996.52	| The pascal SI derived unit of pressure used to quantify internal pressure. Meteorological reports typically state atmospheric pressure in millibars.
3	| T (degC)	| -8.02	| Temperature in Celsius
4	| Tpot (K)	| 265.4	| Temperature in Kelvin
5	| Tdew (degC)	| -8.9	| Temperature in Celsius relative to humidity. Dew Point is a measure of the absolute amount of water in the air, the DP is the temperature at which the air cannot hold all the moisture in it and water condenses.
6	| rh (%)	| 93.3	| Relative Humidity is a measure of how saturated the air is with water vapor, the %RH determines the amount of water contained within collection objects.
7	| VPmax (mbar)	| 3.33	| Saturation vapor pressure
8	| VPact (mbar)	| 3.11	| Vapor pressure
9	| VPdef (mbar)	| 0.22	| Vapor pressure deficit
10	| sh (g/kg)	| 1.94	| Specific humidity
11	| H2OC (mmol/mol)	| 3.12	| Water vapor concentration
12	| rho (g/m ** 3)	| 1307.75	| Airtight
13	| wv (m/s)	| 1.03	| Wind speed
14	| max. wv (m/s)	| 1.75	| Maximum wind speed
15	| wd (deg)	| 152.3	| Wind direction in degrees

## Read the CSV file

If I would like to predict the temperates of **the next 6 hours on the dataset of the nearest 3 hours**, I would have to create windows, each has 18 (3 x 6) data patterns.

The below function will help to do that. `history_size` is the window size (18), target_size is the terperature index I would like to know (6).

The below function is inspired from the above reference link.

```
def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)
```

I will choose the first 300k rows to make them the Train set (about 2083 days). The rest will be the Test set.
```
TRAIN_SPLIT = 300000
```

**Note**

The below model's result might be not good enough. The reason is that in the dataset, it has 14 different features while I will use only 3 features to predict temperature for this project version.

## Predict the time-series data with single variable

I will train the model by only 1 feature `temperature` to predict temperature in the future.

## Plot to observe termperature's fluctuation by **.plot**

I will use `.plot` built in Dataframe (uni_data)

## Tranform data type from Dataframe into Numpy Array

## Normalize the dataset


Note: Mean and Standard deviation should be calculated by the Train set.

From the above step, I have chose the first 300k rows as the Train set.
```
TRAIN_SPLIT = 300000
```
So, I will **calculate mean and std on these 300k rows**, then apply for the whole uni_data.

**Note: I will do it manually, not by using Standard Scaler from the library**

**Further details**

* If the variable A has a Numpy Array type, I will calculate mean and std of A by using A.mean() and A.std()

* Standard Scaler's formula

  $X_{scale} = \frac{X - mean(X)}{std(X)}$

* Keep the variable name as uni_data after scaling

```
# 1. Find mean and std on first 300k rows of uni_data
# 2. Apply scale on uni_data
# 3. Keep variable name (uni_data)
uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()
uni_data = (uni_data - uni_train_mean) / uni_train_std
```

Here, I will predict temperate of the next time step by **the previous 20 time patterns**.

```
univariate_past_history = 20
univariate_future_target = 0

x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)
```

```
print ('Example of data window')
print (x_train_uni[0])
print ('\n Target to be predicted')
print (y_train_uni[0])
```
```
Example of data window
[[-1.99766294]
 [-2.04281897]
 [-2.05439744]
 [-2.0312405 ]
 [-2.02660912]
 [-2.00113649]
 [-1.95134907]
 [-1.95134907]
 [-1.98492663]
 [-2.04513467]
 [-2.08334362]
 [-2.09723778]
 [-2.09376424]
 [-2.09144854]
 [-2.07176515]
 [-2.07176515]
 [-2.07639653]
 [-2.08913285]
 [-2.09260639]
 [-2.10418486]]

 Target to be predicted
-2.1041848598100876
```

```
print ('Example of data window')
print (x_train_uni[1])
print ('\n Target to be predicted')
print (y_train_uni[1])
```

```
Example of data window
[[-2.04281897]
 [-2.05439744]
 [-2.0312405 ]
 [-2.02660912]
 [-2.00113649]
 [-1.95134907]
 [-1.95134907]
 [-1.98492663]
 [-2.04513467]
 [-2.08334362]
 [-2.09723778]
 [-2.09376424]
 [-2.09144854]
 [-2.07176515]
 [-2.07176515]
 [-2.07639653]
 [-2.08913285]
 [-2.09260639]
 [-2.10418486]
 [-2.10418486]]

 Target to be predicted
-2.0949220845536356
```


## Plot the Trained Model


### Baseline

Before designing the model, I will try to predict the current temperature by calculating an average of the previous 20 time steps (a naive approach).



### Simple RNN versus LSTM

I will use tf.data to shuffle the dataset randomly, arrange in batches, and store (data) in a cache memory.

https://stackoverflow.com/questions/53514495/what-does-batch-repeat-and-shuffle-do-with-tensorflow-dataset

```
BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE)
```

I will use SimpleRNN first (a simple model include 1 layer of SimpleRNN and 1 output layer of Dense)

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
```

**Note**

* If there are 2 layers SimpleRNN (or LSTM) simutaneously, it should be added with a parameter return_sequences=True

* Connecting from SimpleRNN layer (or LSTM) to Dense does not require the return_sequences=True
* 
```
# 1. Define architecture for simple_rnn model
# 2. input_shape=x_train_uni.shape[1:]
# 3. Model should have 1 SimpleRNN layer and 1 Dense layer as output
# 4. Fit model using mse loss and adam optimizer
simple_rnn = Sequential()
simple_rnn.add(SimpleRNN(16, input_shape=x_train_uni.shape[1:]))
simple_rnn.add(Dense(1))

simple_rnn.compile(loss='mse', optimizer='adam')
simple_rnn.fit(train_univariate, epochs=10,
                      validation_data=val_univariate)
```

I will take 3 patterns in the validation set (val)univariate) to observe how the SimpleRNN model predicted.



Next, I will replace SimpleRNN layer with LSTM to see if the model could predict more efficiently.

**Note**

The implemention way of LSTM is similar to the one of SimpleRNN.

The model architecture of simple_lstm should be similar to the one of simple_rnn, I will only replace SimpleRNN by LSTM.

```
# 1. Define architecture for simple_lstm model
# 2. Model should be the same like simple_rnn, you just have to replace SimpleRNN with LSTM
# 3. Fit model using mse loss and adam optimizer

simple_lstm = Sequential()
simple_lstm.add(LSTM(16, input_shape=x_train_uni.shape[1:]))
simple_lstm.add(Dense(1))

simple_lstm.compile(loss='mse', optimizer='adam')
simple_lstm.fit(train_univariate, epochs=10,
                      validation_data=val_univariate)
```

Similarly, I will use 3 patterns in the validation set to observer the model's results.


## Predict the Time-Series data with multiple variables

I will predict temperature by 3 features: pressure, temperature and air tight.

```

```

## Plot to observe how the features change over time



Similarly as above, I will scale the data first.

The method is the same but I have many features here so I will need a parameter - axis.


## Prediction model of a new temperature point

In this section, the model will learn and predict a new temperature point based on the known data points (temperature, pressure, airtight).

```
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)
```

The data in the dataset is recorded every 10 minutes. So in 1 hour, it will be recorded 6 times.

I will predict the new temperature point from the dataset of the previous 5 days (5 x 24 x 6) = 720 data points.

Within 1 hour, there will not enough fluctuations so I will get data points separated by 1 hour. So, 720 / 6 = 120 data points.

I would like to predict temperature of the next 12 hours. So, the target is the point separated by 12 x 6 = 72 steps from the current point.

```
past_history = 720
future_target = 72
STEP = 6

x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=True)
```

## Check if the data has exactly 120 points and 3 features

```
print ('Single window of past history : {}'.format(x_train_single[0].shape))
```

```
Single window of past history : (120, 3)
```

```
BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE)
```

## Model with multi-layered LSTM

*Note*

* If there are 2 simultaneous SimpleRNN layers (or LSTM), I will not need the parameter return_sequences=True

* Connecting the SimpleRNN layer (or LSTM) to Dense does not require the return_sequences=True.

### Define architecture for deep_lstm model
```
# 1. Define architecture for deep_lstm model
# 2. input_shape=x_train_single.shape[1:]
# 3. Fit model using mse loss and adam optimizer
deep_lstm = Sequential()
deep_lstm.add(LSTM(16, return_sequences=True,input_shape=x_train_single.shape[1:]))
deep_lstm.add(LSTM(16, return_sequences=True))
deep_lstm.add(LSTM(16))
deep_lstm.add(Dense(1))

deep_lstm.compile(loss='mse', optimizer='adam')
history = deep_lstm.fit(train_data_single, epochs=10,
                      validation_data=val_data_single)
```

```
deep_lstm.summary()
```

```
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm_1 (LSTM)               (None, 120, 16)           1280      
                                                                 
 lstm_2 (LSTM)               (None, 120, 16)           2112      
                                                                 
 lstm_3 (LSTM)               (None, 16)                2112      
                                                                 
 dense_2 (Dense)             (None, 1)                 17        
                                                                 
=================================================================
Total params: 5,521
Trainable params: 5,521
Non-trainable params: 0
_________________________________________________________________
```

### Plot train history



### Predict and draw a prediction line chart
