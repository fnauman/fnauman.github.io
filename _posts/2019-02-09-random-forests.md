One popular class of ensemble machine learning algorithms is random forests that can be used for both classification and regression tasks. The name comes from the use of decision trees ([visual tutorial](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)) as the base estimator. Since decision trees are good at modeling non-linearities and interactions, it makes random forests a popular choice for non-linear datasets where several interactions exist. The final output of random forests is the weighted average over the outputs of individual trees. From a practical standpoint, random forests are extremely popular because:
 - hyperparameter tuning is simpler (as opposed to gradient boosted trees, for example)
 - parallelization is easy since each decision tree is independent

Moreover, data does not need to be normalized or scaled. Being an ensemble method, random forests are robust to outliers and overfitting. Two main sources of randomness in random forests are: bootstrapping (random sampling with replacement) the data for each decision tree with replacement and the choice of a random subset of features for each decision tree. Bootstrapping could be a problem where the data has order, such as in the case of time series.

Despite being an ensemble algorithm, random forest implementations return point estimates with no confidence intervals (see however, [Generalized Random Forests](https://arxiv.org/abs/1610.01271), [Prediction Intervals](https://blog.datadive.net/prediction-intervals-for-random-forests/) ). Another limitation of random forests is related to their particular application to time series: random forests have trouble modeling the trends in time series. Random forests are also hard to interpret: a single decision tree is easy to interpret but an ensemble of 100s or 1000s of decision trees could be hard to make sense of. But interpration of results is assisted by **feature importances** where depending on which features led to the least error in different trees, one can figure out their relative importance. 

One can use random forests in [scikit-learn](https://scikit-learn.org/stable/) like this:
```python
from sklearn.ensemble import RandomForestRegressor
mod = RandomForestRegressor(n_estimators=100)
# Train
mod.fit(X_train, y_train)
# Predict
y_pred = mod.predict(X_test)
```

Hyperparameters that are worth tuning include: `n_estimators`: number of decision trees to use, `max_depth` (depth of each decision tree: too deep trees can lead to overfitting), `min_samples_leaf`: too small a number of samples in the leaf (last) node could also lead to overfitting, `bootstrap`:  setting bootstrap to False is good for time series). 

For a hands-on tutorial on using random forests for a variety of different datasets, I strongly recommend the [machine learning course by Jeremy Howard](http://course18.fast.ai/ml.html). Note that this is not the [deep learning course](https://course.fast.ai/) that is a lot more popular. 
