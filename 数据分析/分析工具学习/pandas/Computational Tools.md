#### Computational tools

###### Statistical functions

Series and `DataFrame` have a method `pct_change()` to compute the percent change over a given number of periods. `Series.cov()` can be used to compute covariance between series. Analogously, `DataFrame.cov()` to compute pairwise covariances among the series in the `DataFrame`, also excluding NA/null values. Correlation may be computed using the `corr()` method. Using the `method` parameter, several methods for computing correlations are provided. All of these are currently computed using pairwise complete observations. A related method `corrwith()` is implemented on `DataFrame` to compute the correlation between like-labeled Series contained in different `DataFrame` objects. The `rank()` method produces a data ranking with ties being assigned the mean of the ranks for the group. `rank()` is also a `DataFrame` method and can rank either the rows or the columns. NaN values are excluded from the ranking. rank optionally takes a parameter `ascending` which by default is true; when false, data is reverse-ranked, with larger values assigned a smaller rank. rank supports different tie-breaking methods, specified with the `method` parameter: average, min, max, first: ranks assigned in the order they appear in the array.

The following table summarizes some other built-in Pandas aggregations:

| Aggregation              | Description                     |
| ------------------------ | ------------------------------- |
| ``count()``              | Total number of items           |
| ``first()``, ``last()``  | First and last item             |
| ``mean()``, ``median()`` | Mean and median                 |
| ``min()``, ``max()``     | Minimum and maximum             |
| ``std()``, ``var()``     | Standard deviation and variance |
| ``mad()``                | Mean absolute deviation         |
| ``prod()``               | Product of all items            |
| ``sum()``                | Sum of all items                |

###### Window Functions

For working with data, a number of window functions are provided for computing common window or rolling statistics. Among these are count, sum, mean, median, correlation, variance, covariance, standard deviation, skewness, and kurtosis. The `rolling()` and `expanding()` functions can be used directly from `DataFrameGroupBy` objects. We work with rolling, expanding and exponentially weighted data through the corresponding objects, Rolling, Expanding and `EWM`.  They all accept the following arguments: window: size of moving window; min_periods: threshold of non-null data points to require; center: boolean, whether to set the labels at the center. They can also be applied to `DataFrame` objects. This is really just syntactic sugar for applying the moving window operator to all of the `DataFrame’s` columns. The `apply()` function takes an extra `func` argument and performs generic rolling computations. The `func` argument should be a single function that produces a single value from an `ndarray` input. 

```python
ser.rolling(window=5, win_type='gaussian').mean(std=0.1)
#For some windowing functions, additional parameters must be specified
```

The inclusion of the interval endpoints in rolling window calculations can be specified with the `closed` parameter

| closed  | Description          | Default for        |
| ------- | -------------------- | ------------------ |
| right   | close right endpoint | time-based windows |
| left    | close left endpoint  |                    |
| both    | close both endpoint  | fixed windows      |
| neither | open endpoints       |                    |

##### Visualization

If the index consists of dates, it calls `gcf().autofmt_xdate()` to try to format the x-axis nicely. Plotting methods allow for a handful of plot styles other than the default line plot. These methods can be provided as the `kind` keyword argument to `plot()`, and include:

| name        | 作用                |
| ----------- | ------------------- |
| `bar, barh` | bar plots           |
| `hist`      | histogram           |
| `box`       | boxplot             |
| `kde`       | density plots       |
| `area`      | area plots          |
| `scatter`   | scatter plots       |
| `hexbin`    | hexagonal bin plots |
| `pie`       | pie plots           |

In addition to these kind s, there are the `DataFrame.hist()`, and `DataFrame.boxplot()` methods, which use a separate interface. `Boxplot` can be drawn calling `Series.plot.box()` and `DataFrame.plot.box()`, or `DataFrame.boxplot()` to visualize the distribution of values within each column. You can create a stratified boxplot using the `by` keyword argument to create groupings. You can create area plots with `Series.plot.area()` and `DataFrame.plot.area()`. Area plots are stacked by default. To produce stacked area plot, each column must be either all positive or all negative values. When input data contains NaN, it will be automatically filled by 0. You can create a pie plot with `DataFrame.plot.pie()` or `Series.plot.pie()`. If your data includes any NaN, they will be automatically filled with 0. A `ValueError` will be raised if there are any negative values in your data. For pie plots it’s best to use square figures, a figure aspect ratio 1. You can create the figure with equal width and height, or force the aspect ratio to be equal after plotting by calling `ax.set_aspect('equal')` on the returned axes object. A legend will be drawn in each pie plots by default, specify `legend=False` to hide it. You can create density plots using the `Series.plot.kde()` and `DataFrame.plot.kde()` methods. 

Lag plots are used to check if a data set or time series is random. Random data should not exhibit any structure in the lag plot. Non-random structure implies that the underlying data are not random. The lag argument may be passed, and when lag=1 the plot is essentially data[:-1] vs. data[1:].
Parallel coordinates is a plotting technique for plotting multivariate data. Parallel coordinates allows one to see clusters in data and to estimate other statistics visually. Using parallel coordinates points are represented as connected line segments. Each vertical line represents one attribute. One set of connected line segments represents one data point. Points that tend to cluster will appear closer together.
Andrews curves allow one to plot multivariate data as a large number of curves that are created using the attributes of samples as coefficients for Fourier series. By coloring these curves differently for each class it is possible to visualize data clustering. Curves belonging to samples of the same class will usually be closer together and form larger structures.
Autocorrelation plots are often used for checking randomness in time series. This is done by computing autocorrelations for data values at varying time lags. If time series is random, such autocorrelations should be near zero for any and all time-lag separations. If time series is non-random then one or more of the autocorrelations will be significantly non-zero. The horizontal lines displayed in the plot correspond to 95% and 99% confidence bands. The dashed line is 99% confidence band.
Bootstrap plots are used to visually assess the uncertainty of a statistic, such as mean, median, midrange, etc. A random subset of a specified size is selected from a data set, the statistic in question is computed for this subset and the process is repeated a specified number of times. Resulting plots and histograms are what constitutes the bootstrap plot.
`RadViz` is a way of visualizing multi-variate data. It is based on a simple spring tension minimization algorithm. Basically you set up a bunch of points in a plane. In our case they are equally spaced on a unit circle. Each point represents a single attribute. You then pretend that each sample in the data set is attached to each of these points by a spring, the stiffness of which is proportional to the numerical value of that attribute. 

A value $x$ is a high-dimensional data point if it is an element of ${R} ^{d}$ We can represent high-dimensional data with a number for each of their dimensions, ${\displaystyle x=\left\{x_{1},x_{2},\ldots ,x_{d}\right\}} $. To visualize them, the Andrews plot defines a finite Fourier series:

${\displaystyle f_{x}(t)={\frac {x_{1}}{\sqrt {2}}}+x_{2}\sin(t)+x_{3}\cos(t)+x_{4}\sin(2t)+x_{5}\cos(2t)+\cdots }$ 
This function is then plotted for ${\displaystyle -\pi <t<\pi } -\pi <t<\pi$ . Thus each data point may be viewed as a line between$ {\displaystyle -\pi }$  and ${\displaystyle \pi }$ . This formula can be thought of as the projection of the data point onto the vector:

${\displaystyle \left({\frac {1}{\sqrt {2}}},\sin(t),\cos(t),\sin(2t),\cos(2t),\ldots \right)}$
If there is structure in the data, it may be visible in the Andrews' curves of the data.