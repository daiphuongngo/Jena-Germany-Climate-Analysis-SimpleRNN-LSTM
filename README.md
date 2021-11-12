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

## Plot to observe termperature's fluctuation by `.plot`

I will ise `.plot` built in Dataframe (uni_data)

