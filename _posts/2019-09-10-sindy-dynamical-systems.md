Sparse Identification of Nonlinear Dynamical systems (SINDy) provides an algorithm to discover governing dynamical equations given a time series **x(t)**. This approach is fundamentally different from most time series models where the focus is either on: smoothing (past data), filtering (current data), forecasting (predicting future data). Here the goal is to uncover the set differential equations that **generated** the time series. It was proposed in this [paper](Paper: https://www.pnas.org/content/113/15/3932) by Steven L. Brunton, Joshua L. Proctor, and J. Nathan Kutz. 

## Algorithm
The three steps in this algorithm are:
 - Time derivative: Perhaps the trickiest part, especially for noisy time series although one can use total variation regularized derivatives for such cases as in [here](https://github.com/stur86/tvregdiff)
 - Basis: Some non-linear basis of the original time series that could be polynomial, trigonometric (see the paper above)
 - Regularized linear regression: Apply LASSO and/or sequential thresholding least squares.
 

## Example
 
 **Notebook** with all the code: [sindy_cubicmodel.ipynb](https://github.com/fnauman/timeseries/blob/master/sindy_cubicmodel.ipynb)
 
Let's first generate some data. Consider the ODE:

$$
\frac{dx_1}{dt} = -x_1^3 - x_2 
$$
$$
\frac{dx_2}{dt} = x_1 - x_2^3 
$$
 
We solve it using the `scipy.odeint` library. Here is how the time series looks like:
![Figure 1](/assets/images/time_series.png)
**Figure 1:**  \\( x_1, x_2 \\) have been evolved to 10 time units with 1000 data points. We will be using the first 800 as the training sample, and the last 200 as the test.
 
### Step 1: Compute the time derivative
In this case, we can compute the time derivative from the data because we know the functional form:
```python
dx1dt = -x1**3 - x2
dx2dt =  x1    - x2**3
```
where x1, x2 have already been computed in the data generation step. Alternatively, one can compute the derivatives using `np.gradient` in numpy or use [total variation denoising](https://en.wikipedia.org/wiki/Total_variation_denoising). Example usage of both is illustrated in the notebook linked above.

### Step 2: Construct polynomial basis
Constructing polynomials from scratch will take some work but we can use scikit-learn to do this for us:
```python
from sklearn.preprocessing import PolynomialFeatures

dum_data = pd.DataFrame(x1, columns=['x1'])
dum_data['x2'] = x2

deg = 3

p = PolynomialFeatures(degree=deg,include_bias=True).fit(dum_data)
xpoly = p.fit_transform(dum_data)
newdf = pd.DataFrame(xpoly, columns = p.get_feature_names(dum_data.columns))

print("Feature names:", list(newdf))
print("Feature array shape:", newdf.shape)
# Output: 
#Feature names: ['1', 'x1', 'x2', 'x1^2', 'x1 x2', 'x2^2', 'x1^3', 'x1^2 x2', 'x1 x2^2', 'x2^3']
#Feature array shape: (1001, 10)
```

### Step 3: Regression
I chose to use Lasso but the original authors used sequential thresholding with MATLAB. 
```python
from sklearn import linear_model
mod = Lasso(alpha=0.0001)
mod
```

The model coefficients resulting from using Lasso are: `
[ 0.         -0.05576026 -0.99852418  0.         -0.         -0.01819165
 -0.50202469 -0.          0.         -0.        ]` where only \\(x_2, x_1^3\\) have amplitudes more than 0.1 (we neglect smaller terms). The SMAPE errors are 1% and 9% for the two ODEs. It seems we have correctly identified the original ODE!!

![Figure 2](/assets/images/dx1dt_fit.png)
![Figure 3](/assets/images/dx2dt_fit.png)
**Figure 2:**  As we can clearly see that the algorithm predictions agree well with the test data.

In the notebook, I also show how to extend this for **forecasting**. I have personally found that SINDy is not so good with series that are too noisy and/or have trends. Both these problems could be addressed using pre-processing. 

