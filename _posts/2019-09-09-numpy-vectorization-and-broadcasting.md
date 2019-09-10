## Vectorization: avoiding loops

Unlike compiled languages like C or Fortran, loops in python are quite slow. When I first started using python several years ago, if I had to compute the average over one axis (y) of a 2D function `v(x,y)`, I would do something like this:
```python
nx, ny = v.shape
ave = np.zeros(nx)
for i in range(nx):
  for j in range(ny):
    ave[i] += v[i, j]
    
ave[i] /= ny
```

Then I got a little smarter and removed one loop:
```python
for i in range(nx):
  ave[i] = np.mean(v[i,:]) 
```

But thanks to **numpy ufuncs** [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/02.03-computation-on-arrays-ufuncs.html), I realized that I do not need any loops at all:
```python
ave = np.mean(v, axis=1)
```
It turns out a whole lot of operations can be **vectorized** using [numpy ufuncs](https://docs.scipy.org/doc/numpy/reference/ufuncs.html). What is even better is that if you are going to repeat the same operation, you can use `partial` from functools to define a numpy ufunc with defaults:
```python
from functools import partial
ave_y = partial(np.mean, axis=1)
ave_y(v) # averages over the 2nd axis
```

## Broadcasting 

Another problem I faced when I first started using python was dealing with arrays with mismatched dimensions. Consider a case where array `x` has shape (10,2) and I want to add another array `y` to it with shape (2). 

```python
x = np.random.randn(10,2) # x.shape = (10,2)
y = np.array([1,2]) # y.shape = (2)
# x+y won't work
x + y[np.newaxis, :] # works!
```

Using **scikit-learn**: The feature array in scikit-learn API is expected to be 2 dimensional. Let's say you only want to do linear regression only on 1 feature (1D = number of samples).

```python
from sklearn.linear_model import Lasso
x = np.random.randn(10)
y = 3*x + 5 + 0.1*np.random.randn(10)
model = Lasso()
model.fit(x,y)
# Returns ValueError: Expected 2D array, got 1D array...
# Reshape your data either using array.reshape(-1, 1) if
# your data has a single feature or array.reshape(1, -1)
```
Scikit-learn throws a helpful error message with a suggested fix: `model.fit(x.reshape(-1,1), y)`. You can also use `x[:, np.newaxis]`.

Alternatively, one might want to use two *feature* arrays of dimension 1 in scikit-learn. Scikit-learn expects dimension 2 feature arrays. 
```python
x1 = np.random.randn(10)
x2 = np.arange(10)
# Efficient way: column_stack
X  = np.column_stack((x1,x2))
# Long, inefficient way
X = np.zeros((x1.shape, 2))
X[:,0] = x1
X[:,1] = x2
```
For more on broadcasting, see [Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow](https://github.com/ageron/handson-ml2/blob/master/tools_numpy.ipynb) and [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html)


## Boolean masking and array dimensions

```python
x = np.linspace(1,10,10)
x>5
# array([False, False, False, False, False,  True,  True,  True,  True, True])
x[x>5] # array([ 6.,  7.,  8.,  9., 10.])
```
A conditional statement `x>5` generates a boolean array of the same dimension as the original array. When we use `x[x>5]`, only the values satisfying the conditional statement are returned. In many applications, it is not a problem but if you have code that breaks if the array dimensions don't match anymore, you want to simply multiply the boolean array:

```python
x * (x>5)
# array([ 0.,  0.,  0.,  0.,  0.,  6.,  7.,  8.,  9., 10.])
```

**xarray**: A lot of these vectorization and broadcasting operations are made easy by [xarray](http://xarray.pydata.org/en/stable/quick-overview.html) where as a bonus, parallelism is included using [dask](https://dask.org/). A good tutorial on the utility of xarray is [here](https://rabernat.github.io/research_computing/xarray.html).
