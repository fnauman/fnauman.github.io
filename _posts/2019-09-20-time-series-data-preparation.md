When forecasting time series using neural networks, one has to first transform the data into the required form. For regression algorithms such as random forests or linear regression, one can transform univariate time series relatively easily:

```python
import numpy as np
omega = 2 * np.pi * np.linspace(-1,1,100) 
T = 1
tseries = np.sin(omega * T) # 100 snapshots

feature_size = int(0.8 * tseries.shape[0])
#target_size = int(0.2 * tseries.shape[0])

X, y = tseries[:feature_size].reshape(-1,1), tseries[feature_size:]
```

This is not recommended as no validation and test sets have been created. Splitting time series data for cross validation and testing can be tricky as opposed to other data: one has to be careful in not violating temporal causality. Common strategies include a [rolling/sliding (fixed size) window; or an expanding window](https://stats.stackexchange.com/questions/326228/cross-validation-with-time-series). The [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) function in scikit-learn for cross validation uses the expanding window. For an implementation of the fixed size rolling window, see the [code here](https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/). Another relevant issue with time series data is of using the appropriate metrics. [Marios Michailidis](https://www.h2o.ai/blog/regression-metrics-guide/) discusses the pros and cons of using various popular metrics for time series.

## Input data shape for RNNs

For neural networks like Recurrent Neural Networks/Long Short Term Memory the data has to be in the shape:
```python
X = (n_samples, input_sequence_length, n_input_features)
y = (n_samples, output_sequence_length, n_output_features)
```

For large datasets, it is convenient to input data only "one batch at a time" (`n_samples -> batch_size`). The sequence length is often called the [lookback time](https://stackoverflow.com/questions/45012992/how-to-prepare-data-for-lstm-when-using-multiple-time-series-of-different-length) or [input steps](https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/). For univariate time series forecasting problem, `n_input_features = n_output_features = 1`. For the **target vector**, typically `output_sequence_length=1` (single point forecasting) is chosen but one can recursively forecast several points to the future. The distinction between **many-to-one** and **many-to-many** can be confusing (see [Andrej Karpathy's blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)). See also the  [notebook](https://github.com/ageron/handson-ml2/blob/master/15_processing_sequences_using_rnns_and_cnns.ipynb) by [Aurélien Géron](https://www.amazon.com/gp/product/B07XGF2G87/ref=dbs_a_def_rwt_hsch_vapi_tkin_p1_i0) for tensorflow 2.0/keras examples. 

**seq2seq**: For making predictions from variable input sequence lengths to (entire) output sequence lengths in one go (as opposed to recursive single step forecasting), one can use sequence-to-sequence (encoder-decoder) models [see this paper by Sutskever et al. 2014](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf).

## Preparing data using numpy/scipy

Converting a given time series data to the required input format for RNNs/LSTMs is not difficult with numpy. Several good implementations exist, for example: [here](https://github.com/pipidog/DeepTimeSeries/blob/master/DeepTimeSeries/utils.py).

**Hankel** matrix is a matrix with a specific structure:
```
[ x(0)  x(1)    ...    x(c-1) ]
[ x(1)  x(2)    ...      x(c) ]
[               ...           ] 
[ x(r-1)  x(r)  ...  x(r+c-2) ]
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

**Shift/Delay**: What if we want to introduce a shift in the array? That is instead of taking continuous values of the time series `0, 1, ..., n-1`, we want something like: `0, delay, 2*delay, ..., n-1`.  It is trivial to do with numpy's array indexing: `array[::delay]`:
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

We want to use this format for time series since each row (**sample**) can be thought of as representing a sequence that can be broken into a feature and target set. Consider this example:
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

`y_train` contains a single point because we specified the target size to be one. When this is fed into a RNN, it trains on `x_train` with single point labels `y_train`. The predictions from such a model generate a forecast that can be compared with the target test sequence (`y_test`) to see how good the RNN is. 

**Shuffling the samples**: Feeding a neural network data that represents similar time points can lead to overfitting. In non-sequential data, random shuffling helps to avoid overfitting. A naive random shuffling of time series can lead to problems since temporal data has *order*: an event at time `t` cannot precede an event at time `t-100`. But once we have already broken our time series into train and test sequences, we can shuffle the *samples/rows* without violating the temporal order since each target vector contains temporal points after the corresponding points in the feature vector.

Numpy provides convenient functions to do the shuffling in one line. `np.random.shuffle` shuffles the data in-place, which might be a good idea for large arrays. `np.random.permutation` on the other hand creates a new array.  One can define a function to do this:
```python
def shuffle(X, y, seed=123):
    np.random.seed(seed)
    idxs = np.arange(y.shape[0])
    random_idxs = np.random.permutation(idxs)
    return X[random_idxs], y[random_idxs]
```
Some gotchas with `np.random.permutation`: 
 - Repeated calls require setting the seed [again](https://stackoverflow.com/questions/47742622/np-random-permutation-with-seed/47742662#47742662) 
 - `np.random.permutation` by default only permutes the zeroth (`axis=0`) dimension of a numpy array, which is fine in the case above for 1D array. To shuffle another axis, one can use: `np.apply_along_axis(np.random.permutation, 1, x)` for shuffling across the first (`axis=1`) dimension.

**Batching**: Feeding neural networks an entire data set is not recommended because:
 - Feeding entire dataset in one go can be computationally prohibitive.
 - Breaking the data into small batches can help avoid overfitting as models are more likely to see different patterns. 
 
 For more on batch learning and stochastic gradient descent, see:
  - [Efficient backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) by Yann LeCun, Leon Bottou, Genevieve B. Orr, Klaus-Robert Müller.
  - [Chapter 8](http://www.deeplearningbook.org/contents/optimization.html) of Deep learning by Ian Goodfellow, Yoshua Bengio and Aaron Courville.

Batching can be done in `keras` during the fitting step: `model.fit(X_train, y_train, ..., batch_size=32)` or one could create a data generator object at the input step ([example](https://www.kaggle.com/ezietsman/simple-keras-model-with-data-generator)). Tensorflow has its own dataset API that can be used for time series following their [tutorial](https://www.tensorflow.org/beta/guide/data#time_series_windowing)

 



