When modeling time series using neural networks, the first issue one has to deal with is the data formatting. For machine learning algorithms like random forests or linear regression, one can transform univariate time series like this:

```python
import numpy as np
omega = 2 * np.pi * np.linspace(-1,1,100) 
T = 1
tseries = np.sin(omega * T) # 100 snapshots

feature_size = int(0.8 * tseries.shape[0])
#target_size = int(0.2 * tseries.shape[0])

X, y = tseries[:feature_size].reshape(-1,1), tseries[feature_size:]
```

This, however, is not recommended as no validation and test sets have been created. Splitting time series data for cross validation and testing even for machine learning models is not the same as other data: one has to choose either a [rolling/sliding (fixed size) window; or an expanding window](https://stats.stackexchange.com/questions/326228/cross-validation-with-time-series). The [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) function in scikit-learn for cross validation uses expanding window. For an implementation of fixed size rolling window, see the [code here](https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/). Another relevant issue with time series data is of using appropriate metrics. [Marios Michailidis](https://www.h2o.ai/blog/regression-metrics-guide/) discusses the pros and cons of using various popular metrics for time series.

## Input data shape for RNNs

For neural networks like Recurrent Neural Networks/Long Short Term Memory the data has to be in the shape:
```python
X = (n_samples, input_sequence_length, n_input_features)
y = (n_samples, output_sequence_length, n_output_features)
```

For large datasets, `n_samples -> batch_size` while sequence length is often called the [lookback time](https://stackoverflow.com/questions/45012992/how-to-prepare-data-for-lstm-when-using-multiple-time-series-of-different-length) or [input steps](https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/). For univariate time series forecasting problem, `n_input_features = n_output_features = 1`. For the **target vector**, typically `output_sequence_length=1` (single point forecasting) but one can recursively forecast several points to the future. The distinction between **many-to-one** and **many-to-many** can be confusing (see [Andrej Karpathy's blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)). See also this [notebook](https://github.com/ageron/handson-ml2/blob/master/15_processing_sequences_using_rnns_and_cnns.ipynb) by [Aurélien Géron](https://www.amazon.com/gp/product/B07XGF2G87/ref=dbs_a_def_rwt_hsch_vapi_tkin_p1_i0) for tensorflow 2.0/keras examples. 

**seq2seq**: For making predictions from variable input sequence lengths to entire output sequence lengths in one go (instead of recursive single step forecasting), one can use sequence-to-sequence (encoder-decoder) models [see this paper by Sutskever et al. 2014](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf).

## Preparing data using numpy/scipy

Converting a given time series data to the required input format for RNNs/LSTMs is not difficult with numpy. Several good implementations exist, for example: [here](https://github.com/pipidog/DeepTimeSeries/blob/master/DeepTimeSeries/utils.py).

**Hankel** matrix is a matrix with a specific structure:
```
[ x(0)  x(1) ... x(c-1)     ]
[ x(1)  x(1) ... x(c)       ]
[ ...                       ] 
[ x(r-2)  x(r-1) ... x(c-1) ]
[ x(r-1)  x(r) ... x(r+c-1) ]
```

where r, c represent number of rows, columns. Using `scipy.hankel`, one can quickly generate such a matrix:
```python
from scipy.linalg import hankel
hankel(np.arange(4))

# Output
array([[0, 1, 2, 3],
       [1, 2, 3, 0],
       [2, 3, 0, 0],
       [3, 0, 0, 0]])
```

This matrix contains zeros in the lower triangle. If we want to restrict to **non-zero values**, we need to use:
```python
arr = np.arange(4)
nrows = 3
hankel(c=arr[:nrows], r=arr[nrows-1:])
# Output
array([[0, 1],
       [1, 2],
       [2, 3]])
```

**Shift/Delay**: What if we want to introduce a shift in the array? That is instead of taking continuous values of the time series `0, 1, ..., n-1`, we want something like: `0, shift, 2*shift, ..., n-1`.  It is trivial to do with numpy's array indexing: `array[::shift]`:
```python
arr = np.arange(8)
nrows = 3
delay = 2
arr = arr[::delay]
hankel(c=arr[:nrows], r=arr[nrows-1:])
# Output
array([[0, 2],
       [2, 4],
       [4, 6]])
```

We want to use this format for time series since each row can be thought of as representing a sequence that can be broken into a feature and target set. Consider this example:
```python
x = np.arange(21)
x_hank = create_hankel(x, nrows = 4)
x_hank
# Output
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17],
       [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18],
       [ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19],
       [ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20]])
```

**Feature-target split**: Now break this into features and targets:
```python
target_size = 1
x_train, y_train = x_hank[:, :-target_size], x_hank[:, -target_size:]
print(x_train, y_train)
# Output
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16],
       [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17],
       [ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18],
       [ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19]]) # x_train
array([17, 18, 19, 20]) # y_train

```

`y_train` contains a single point because we specified the target size to be one. When this is fed into a RNN, it takes in the entire `x_train` sequence and forecasts a single point that can be compared with `y_train` to see how good the RNN is. We can forecast multiple steps recursively using RNNs.





