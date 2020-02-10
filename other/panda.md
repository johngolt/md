#### Indexing and selecting data

Object selection has had a number of user-requested additions in order to support more explicit location based indexing. Pandas now supports three types of multi-axis indexing.

- `.loc` is primarily label based, but may also be used with a boolean array. `.loc` will raise `KeyError` when the items are not found. Allowed inputs are: A single label; A list or array of labels; A slice object with labels 'a':'f', Note that contrary to usual python slices, both the start and the stop are included, when present in the index; A boolean array; A callable function with one argument and that returns valid output for indexing.

- `.iloc` is primarily integer position based, but may also be used with a boolean array. `.iloc` will raise `IndexError` if a requested indexer is out-of-bounds, except slice indexers which allow out-of-bounds indexing. Allowed inputs are: An integer; A list or array of integers; A slice object with ints; A boolean array; A callable function with one argument and that returns valid output for indexing.

###### Slicing ranges

You may access an index on a Series or column on a DataFrame directly as an attribute. You can also assign a dict to a row of a DataFrame. the semantics of slicing using the [] operator. With Series, the syntax works exactly as with an ndarray, returning a slice of the values and the corresponding labels. With DataFrame, slicing inside of [] slices the rows. This is provided largely as a convenience since it is such a common operation.

###### Selection by label

When using .loc with slices, if both the start and the stop labels are present in the index, then elements located between the two (including them) are returned:
If at least one of the two is absent, but the index is sorted, and can be compared against start and stop labels, then slicing will still work as expected, by selecting labels which rank between the two:

```python
s = pd.Series(list('abcde'), index=[0, 3, 2, 5, 4])
s.sort_index().loc[1:6]
```

`.loc`, `.iloc`, and also `[]` indexing can accept a `callable` as indexer. The `callable` must be a function with one argument (the calling Series or DataFrame) that returns valid output for indexing.

using `.loc[list-of-labels]` work as long as at least 1 of the keys was found. This behavior is deprecated and will show a warning message pointing to this section. The recommended alternative is to use `.reindex`.

```python
s = pd.Series([1, 2, 3])
s.reindex([1, 2, 3]) # Having a duplicated index will raise error for a .reindex()
s.loc[s.index.intersection([1,2,3])]  # select only valid keys
```

###### Selecting random samples

A random selection of rows or columns from a Series or DataFrame with the `sample()` method. The method will sample rows by default, and accepts a specific number of rows/columns to return, or a fraction of rows.

By default, sample will return each row at most once, but one can also sample with replacement using the `replace` option. By default, each row has an equal probability of being selected, but if you want rows to have different probabilities, you can pass the `sample` function sampling weights as `weights`. Missing values will be treated as a weight of zero, and inf values are not allowed. If weights do not sum to 1, they will be re-normalized by dividing all weights by the sum of the weights. When applied to a DataFrame, you can use a column of the DataFrame as sampling weights by simply passing the name of the column as a string. sample also allows users to sample columns instead of rows using the `axis` argument. Finally, one can also set a seed for sample’s random number generator using the `random_state` argument. 

###### Boolean indexing

If you only want to access a scalar value, the fastest way is to use the `at` and `iat` methods, which are implemented on all of the data structures. Similarly to `loc`, at provides label based scalar lookups, while, `iat` provides integer based lookups analogously to `iloc`.

Another common operation is the use of boolean vectors to filter the data. The operators are: `|` for or, `&` for and, and `~` for not. These must be grouped by using parentheses, since by default Python will evaluate an expression such as `df.A > 2 & df.B < 3` as `df.A > (2 & df.B) < 3`. Using a boolean vector to index a Series works exactly as in a NumPy. You may select rows from a DataFrame using a boolean vector the same length as the DataFrame’s index.

Consider the `isin()` method of Series, which returns a boolean vector that is true wherever the Series elements exist in the passed list. This allows you to select rows where one or more columns have values you want. DataFrame also has an `isin()` method. When calling `isin`, pass a set of values as either an array or `dict`. If values is an array, `isin` returns a DataFrame of booleans that is the same shape as the original DataFrame, with True wherever the element is in the sequence of values. Oftentimes you’ll want to match certain values with certain columns. Just make values a `dict` where the key is the column, and the value is a list of items you want to check for. Combine DataFrame’s `isin` with the `any()` and `all()` methods to quickly select subsets of your data that meet a given criteria. 

###### The `where()` Method and Masking

Selecting values from a Series with a boolean vector generally returns a subset of the data. To guarantee that selection output has the same shape as the original data, you can use the `where` method in Series and DataFrame. In addition, `where` takes an optional `other` argument for replacement of values where the condition is `False`, in the returned copy. By default, `where` returns a modified copy of the data. There is an optional parameter `inplace` so that the original data can be modified without creating a copy. `Where` can also accept `axis` and `level` parameters to align the input when performing the `where`. `Where` can accept a callable as condition and other arguments. The function must be with one argument (the calling Series or DataFrame) and that returns valid output as condition and other argument. `mask()` is the inverse boolean operation of `where`.

```python
df2.where(df2 > 0, df2['A'], axis='index')
df3.where(lambda x: x > 4, lambda x: x + 10)
```

###### The `query()` Method

```python
df = pd.DataFrame(np.random.rand(n, 3), columns=list('abc'))
df.query('(a < b) & (b < c)')
# If instead you don’t want to or cannot name your index, you can use the name index in your query expression.
df.query('index < b < c')
df.query('[1, 2] in c')
```

If the name of your index overlaps with a column name, the column name is given precedence. You can still use the index in a query expression by using the special identifier ‘index’. You can also use the `levels` of a DataFrame with a MultiIndex as if they were columns in the frame. If the `levels` of the MultiIndex are unnamed, you can refer to them using special names. The convention is `ilevel_0`, which means “index level 0” for the 0th level of the index. `query()` also supports special use of Python’s `in` and `not in` comparison operators, providing a succinct syntax for calling the `isin` method of a Series or DataFrame. Comparing a list of values to a column using `==/!=` similarly to `in/not in`.

##### Duplicate data

If you want to identify and remove duplicate rows in a DataFrame, there are two methods that will help: `duplicated` and `drop_duplicates`. Each takes as an argument the columns to use to identify duplicated rows. `duplicated` returns a boolean vector whose length is the number of rows, and which indicates whether a row is duplicated.
`drop_duplicates` removes duplicate rows. By default, the first observed row of a duplicate set is considered unique, but each method has a `keep` parameter to specify targets to be kept.

- keep='first' (default): mark / drop duplicates except for the first occurrence.
- keep='last': mark / drop duplicates except for the last occurrence.
- keep=False: mark / drop all duplicates.

Also, you can pass a list of columns to identify duplications. Each of Series or DataFrame have a `get` method which can return a default value.

Indexes are “mostly immutable”, but it is possible to set and change their metadata, like the index name or, for MultiIndex, levels and codes. You can use the `rename, set_names, set_levels`, and `set_codes` to set these attributes directly. They default to returning a copy; however, you can specify `inplace=True` to have the data change in place. The two main operations are union (|) and intersection (&). These can be directly called as instance methods or used via overloaded operators. Difference is provided via the `.difference` method. When performing `Index.union()` between indexes with different dtypes, the indexes must be cast to a common dtype. Typically, though not always, this is object dtype. The exception is when performing a union between integer and float data. In this case, the integer values are converted to float. DataFrame has a `set_index()` method which takes a column name or a list of column names. To create a new, re-indexed DataFrame. As a convenience, there is a new function on DataFrame called `reset_index()` which transfers the index values into the DataFrame’s columns and sets a simple integer index. This is the inverse operation of `set_index()`. You can use the `level` keyword to remove only a portion of the index.

#### Merge, join and concatenate

###### Concatenating objects

The `concat()` function does all of the heavy lifting of performing concatenation operations along an axis while performing optional set logic (union or intersection) of the indexes (if any) on the other axes. Note that I say “if any” because there is only a single possible axis of concatenation for Series.

```python
 result = pd.concat(frames, keys=['x', 'y', 'z'])
```

![](../picture/1/152.png)

When gluing together multiple DataFrames, you have a choice of how to handle the other axes. This can be done in the following two ways: Take the union of them all, join='outer'. Take the intersection, join='inner'.

```python
result = pd.concat([df1, df4], axis=1).reindex(df1.index)
```

![](../picture/1/153.png)

A useful shortcut to `concat()` are the `append()` instance methods on Series and DataFrame. These methods actually predated `concat`. They concatenate along `axis=0`, namely the index. For `DataFrame` objects which don’t have a meaningful index, you may wish to append them and ignore the fact that they may have overlapping indexes. To do this, use the `ignore_index` argument. You can concatenate a mix of `Series` and `DataFrame` objects. The `Series` will be transformed to `DataFrame` with the column name as the name of the `Series`. If unnamed `Series` are passed they will be numbered consecutively. Passing `ignore_index=True` will drop all name references. A fairly common use of the `keys` argument is to override the column names when creating a new `DataFrame` based on existing `Series`. Notice how the default behaviour consists on letting the resulting `DataFrame` inherit the parent `Series`’ name, when these existed. Through the `keys` argument we can override the existing column names. You can also pass a dict to `concat` in which case the dict keys will be used for the `keys` argument (unless other keys are specified)

pandas provides a single function, `merge()`, as the entry point for all standard database join operations between DataFrame or named Series objects. If left is a DataFrame or named Series and right is a subclass of DataFrame, the return type will still be DataFrame. `merge` is a function in the pandas namespace, and it is also available as a DataFrame instance method `merge()`, with the calling DataFrame being implicitly considered the left object in the join. The related `join()` method, uses merge internally for the index-on-index and column(s)-on-index join. If you are joining on index only, you may wish to use `DataFrame.join` to save yourself some typing.
 There are several cases to consider which are very important to understand: one-to-one joins, many-to-one joins, many-to-many joins.
In SQL / standard relational algebra, if a key combination appears more than once in both tables, the resulting table will have the Cartesian product of the associated data.
The `how` argument to merge specifies how to determine which keys are to be included in the resulting table. If a key combination does not appear in either the left or right tables, the values in the joined table will be NA.

Users can use the `validate` argument to automatically check whether there are unexpected duplicates in their merge keys. Key uniqueness is checked before merge operations and so should protect against memory overflows. Checking key uniqueness is also a good way to ensure user data structures are as expected.

```python
result = pd.merge(left, right, on='B', how='outer', validate="one_to_one")
pd.merge(left, right, on='B', how='outer', validate="one_to_many")
```

You can join a singly-indexed DataFrame with a level of a MultiIndexed DataFrame. The level will match on the name of the index of the singly-indexed frame against a level name of the MultiIndexed frame. Strings passed as the `on, left_on`, and `right_on` parameters may refer to either column names or index level names. This enables merging DataFrame instances on a combination of index levels and columns without resetting indexes. The merge `suffixes` argument takes a tuple of list of strings to append to overlapping column names in the input DataFrames to disambiguate the result columns.

#### Reshaping and pivot tables

![](../picture/1/154.png)

`pivot()` will error with a `ValueError`: Index contains duplicate entries, cannot reshape if the index/column pair is not unique. In this case, consider using `pivot_table()` which is a generalization of pivot that can handle duplicate values for one index/column pair.

If the columns have a `MultiIndex`, you can choose which level to stack. The stacked level becomes the new lowest level in a `MultiIndex` on the columns. With a “stacked” `DataFrame` or `Series` (having a `MultiIndex` as the `index`), the inverse operation of `stack` is `unstack`, which by default unstacks the **last level**:

![](../picture/1/155.png)

![](../picture/1/156.png)

If the indexes have names, you can use the level names instead of specifying the level numbers. Notice that the `stack` and `unstack` methods implicitly sort the index levels involved. Hence a call to `stack` and then `unstack`, or vice versa, will result in a **sorted** copy of the original `DataFrame` or `Series`. You may also stack or unstack more than one level at a time by passing a list of levels, in which case the end result is as if each level in the list were processed individually. The list of levels can contain either level names or level numbers. Unstacking can result in missing values if subgroups do not have the same set of labels. By default, missing values will be replaced with the default fill value for that data type, `NaN` for float.

![](../picture/1/157.png)

![](../picture/1/158.png)

The top-level `melt()` function and the corresponding `DataFrame.melt()` are useful to massage a DataFrame into a format where one or more columns are identifier variables, while all other columns, considered measured variables, are “unpivoted” to the row axis, leaving just two non-identifier columns, “variable” and “value”. The names of those columns can be customized by supplying the `var_name` and `value_name` parameters.

##### Computational tools

###### Statistical functions

Series and DataFrame have a method `pct_change()` to compute the percent change over a given number of periods. `Series.cov()` can be used to compute covariance between series. Analogously, `DataFrame.cov()` to compute pairwise covariances among the series in the DataFrame, also excluding NA/null values. Correlation may be computed using the `corr()` method. Using the `method` parameter, several methods for computing correlations are provided. All of these are currently computed using pairwise complete observations. A related method `corrwith()` is implemented on DataFrame to compute the correlation between like-labeled Series contained in different DataFrame objects. The `rank()` method produces a data ranking with ties being assigned the mean of the ranks for the group. `rank()` is also a DataFrame method and can rank either the rows (axis=0) or the columns (axis=1). NaN values are excluded from the ranking. rank optionally takes a parameter `ascending` which by default is true; when false, data is reverse-ranked, with larger values assigned a smaller rank. rank supports different tie-breaking methods, specified with the `method` parameter: average, min, max, first: ranks assigned in the order they appear in the array.

###### Window Functions

For working with data, a number of window functions are provided for computing common window or rolling statistics. Among these are count, sum, mean, median, correlation, variance, covariance, standard deviation, skewness, and kurtosis. The `rolling()` and `expanding()` functions can be used directly from DataFrameGroupBy objects. We work with rolling, expanding and exponentially weighted data through the corresponding objects, Rolling, Expanding and EWM. Generally these methods all have the same interface. They all accept the following arguments:

- window: size of moving window
- min_periods: threshold of non-null data points to require
- center: boolean, whether to set the labels at the center

We can then call methods on these rolling objects. These return like-indexed objects. They can also be applied to DataFrame objects. This is really just syntactic sugar for applying the moving window operator to all of the DataFrame’s columns.

##### Visualization

If the index consists of dates, it calls `gcf().autofmt_xdate()` to try to format the x-axis nicely. Plotting methods allow for a handful of plot styles other than the default line plot. These methods can be provided as the `kind` keyword argument to plot(), and include:

- ‘bar’ or ‘barh’ for bar plots
- ‘hist’ for histogram
- ‘box’ for boxplot
- ‘kde’ or ‘density’ for density plots
- ‘area’ for area plots
- ‘scatter’ for scatter plots
- ‘hexbin’ for hexagonal bin plots
- ‘pie’ for pie plots

In addition to these kind s, there are the `DataFrame.hist()`, and `DataFrame.boxplot()` methods, which use a separate interface. Boxplot can be drawn calling `Series.plot.box()` and `DataFrame.plot.box()`, or `DataFrame.boxplot()` to visualize the distribution of values within each column. You can create a stratified boxplot using the by keyword argument to create groupings. You can create area plots with `Series.plot.area()` and `DataFrame.plot.area()`. Area plots are stacked by default. To produce stacked area plot, each column must be either all positive or all negative values. When input data contains NaN, it will be automatically filled by 0. Scatter plot can be drawn by using the `DataFrame.plot.scatter()` method. Scatter plot requires numeric columns for the x and y axes. These can be specified by the `x` and `y` keywords. You can create a pie plot with `DataFrame.plot.pie()` or `Series.plot.pie()`. If your data includes any NaN, they will be automatically filled with 0. A ValueError will be raised if there are any negative values in your data. For pie plots it’s best to use square figures, i.e. a figure aspect ratio 1. You can create the figure with equal width and height, or force the aspect ratio to be equal after plotting by calling `ax.set_aspect('equal')` on the returned axes object. Note that pie plot with DataFrame requires that you either specify a target column by the y argument or `subplots=True`. When y is specified, pie plot of selected column will be drawn. If `subplots=True` is specified, pie plots for each column are drawn as subplots. A legend will be drawn in each pie plots by default; specify `legend=False` to hide it. You can create density plots using the `Series.plot.kde()` and `DataFrame.plot.kde()` methods.

Finally, there are several plotting functions in `pandas.plotting` that take a Series or DataFrame as an argument. These include: Scatter Matrix, Andrews Curves, Parallel Coordinates, Lag Plot, Autocorrelation Plot, Bootstrap Plot, RadViz

Lag plots are used to check if a data set or time series is random. Random data should not exhibit any structure in the lag plot. Non-random structure implies that the underlying data are not random. The lag argument may be passed, and when lag=1 the plot is essentially data[:-1] vs. data[1:].
Parallel coordinates is a plotting technique for plotting multivariate data. Parallel coordinates allows one to see clusters in data and to estimate other statistics visually. Using parallel coordinates points are represented as connected line segments. Each vertical line represents one attribute. One set of connected line segments represents one data point. Points that tend to cluster will appear closer together.
Andrews curves allow one to plot multivariate data as a large number of curves that are created using the attributes of samples as coefficients for Fourier series. By coloring these curves differently for each class it is possible to visualize data clustering. Curves belonging to samples of the same class will usually be closer together and form larger structures.
Autocorrelation plots are often used for checking randomness in time series. This is done by computing autocorrelations for data values at varying time lags. If time series is random, such autocorrelations should be near zero for any and all time-lag separations. If time series is non-random then one or more of the autocorrelations will be significantly non-zero. The horizontal lines displayed in the plot correspond to 95% and 99% confidence bands. The dashed line is 99% confidence band.
Bootstrap plots are used to visually assess the uncertainty of a statistic, such as mean, median, midrange, etc. A random subset of a specified size is selected from a data set, the statistic in question is computed for this subset and the process is repeated a specified number of times. Resulting plots and histograms are what constitutes the bootstrap plot.
RadViz is a way of visualizing multi-variate data. It is based on a simple spring tension minimization algorithm. Basically you set up a bunch of points in a plane. In our case they are equally spaced on a unit circle. Each point represents a single attribute. You then pretend that each sample in the data set is attached to each of these points by a spring, the stiffness of which is proportional to the numerical value of that attribute (they are normalized to unit interval). The point in the plane, where our sample settles to (where the forces acting on our sample are at an equilibrium) is where a dot representing our sample will be drawn. Depending on which class that sample belongs it will be colored differently. 

A value $x$ is a high-dimensional datapoint if it is an element of ${R} ^{d}$ We can represent high-dimensional data with a number for each of their dimensions, ${\displaystyle x=\left\{x_{1},x_{2},\ldots ,x_{d}\right\}} $. To visualize them, the Andrews plot defines a finite Fourier series:

${\displaystyle f_{x}(t)={\frac {x_{1}}{\sqrt {2}}}+x_{2}\sin(t)+x_{3}\cos(t)+x_{4}\sin(2t)+x_{5}\cos(2t)+\cdots }$ 
This function is then plotted for ${\displaystyle -\pi <t<\pi } -\pi <t<\pi$ . Thus each data point may be viewed as a line between$ {\displaystyle -\pi }$  and ${\displaystyle \pi }$ . This formula can be thought of as the projection of the data point onto the vector:

${\displaystyle \left({\frac {1}{\sqrt {2}}},\sin(t),\cos(t),\sin(2t),\cos(2t),\ldots \right)}$
If there is structure in the data, it may be visible in the Andrews' curves of the data.

##### Group By

By “group by” we are referring to a process involving one or more of the following steps: Splitting the data into groups based on some criteria; Applying a function to each group independently; Combining the results into a data structure.
In the apply step, we might wish to do one of the following:

- Aggregation: compute a summary statistic for each group. 
- Transformation: perform some group-specific computations and return a like-indexed object. 
- Filtration: discard some groups, according to a group-wise computation that evaluates True or False. Some examples.

###### Splitting

pandas objects can be split on any of their axes. The abstract definition of grouping is to provide a mapping of labels to group names.  The mapping can be specified many different ways:

- A Python function, to be called on each of the axis labels.
- A list or NumPy array of the same length as the selected axis.
- A dict or Series, providing a label -> group name mapping.
- For DataFrame objects, a string indicating a column to be used to group. Of course df.groupby('A') is just syntactic sugar for df.groupby(df['A']), but it makes life simpler.
- For DataFrame objects, a string indicating an index level to be used to group.
  A list of any of the above things.

pandas Index objects support duplicate values. If a non-unique index is used as the group key in a groupby operation, all values for the same index value will be considered to be in one group and thus the output of aggregation functions will only contain unique index values
Note that no splitting occurs until it’s needed. Creating the GroupBy object only verifies that you’ve passed a valid mapping. By default the group keys are sorted during the groupby operation. You may however pass `sort=False` for potential speedups. The `groups` attribute is a dict whose keys are the computed unique groups and corresponding values being the axis labels belonging to each group. With hierarchically-indexed data, it’s quite natural to group by one of the levels of the hierarchy. If the MultiIndex has names specified, these can be passed instead of the level number. A DataFrame may be grouped by a combination of columns and index levels by specifying the column names as strings and the index `levels` as `pd.Grouper` objects. Once you have created the GroupBy object from a DataFrame, you might want to do something different for each of the columns. Thus, using [] similar to getting a column from a DataFrame, you can do

###### Aggregation

Once the GroupBy object has been created, several methods are available to perform a computation on the grouped data. An obvious one is aggregation via the `aggregate()` or equivalently `agg()` method. Any function which reduces a Series to a scalar value is an aggregation function and will work, a trivial example is `df.groupby('A').agg(lambda ser: 1)`. With grouped Series you can also pass a list or dict of functions to do aggregation with, outputting a DataFrame. On a grouped DataFrame, you can pass a list of functions to apply to each column, which produces an aggregated result with a hierarchical index. To support column-specific aggregation with control over the output column names, pandas accepts the special syntax in GroupBy.agg(), known as “named aggregation”, where The keywords are the output column names, The values are tuples whose first element is the column to select and the second element is the aggregation to apply to that column. Pandas provides the pandas.NamedAgg namedtuple with the fields ['column', 'aggfunc'] to make it clearer what the arguments are. 

```python
 animals.groupby("kind").agg(min_height=pd.NamedAgg(column='height', aggfunc='min'),max_height=pd.NamedAgg(column='height', aggfunc='max'),
  average_weight=pd.NamedAgg(column='weight', aggfunc=np.mean)
# pandas.NamedAgg is just a namedtuple. Plain tuples are allowed as well.
animals.groupby("kind").agg(min_height=('height', 'min'),max_height=('height', 'max'),average_weight=('weight', np.mean),
```

Additional keyword arguments are not passed through to the aggregation functions. Only pairs of (column, aggfunc) should be passed as. If your aggregation functions requires additional arguments, partially apply them with `functools.partial()`

The transform method returns an object that is indexed the same as the one being grouped. The transform function must: Return a result that is either the same size as the group chunk or broadcastable to the size of the group chunk. Operate column-by-column on the group chunk. The transform is applied to the first group chunk using chunk.apply. Not perform in-place operations on the group chunk. Group chunks should be treated as immutable, and changes to a group chunk may produce unexpected results. For example, when using fillna, inplace must be False. Transformation functions that have lower dimension outputs are broadcast to match the shape of the input array. Working with the resample, expanding or rolling operations on the groupby level used to require the application of helper functions. However, now it is possible to use `resample(), expanding()` and `rolling()` as methods on groupbys.
The filter method returns a subset of the original object. The argument of filter must be a function that, applied to the group as a whole, returns True or False.

Some operations on the grouped data might not fit into either the aggregate or transform categories. Or, you may simply want GroupBy to infer how to combine the results. For these, use the apply function, which can be substituted for both aggregate and transform in many standard use cases. However, apply can handle some exceptional use cases, apply on a Series can operate on a returned value from the applied function, that is itself a series, and possibly upcast the result to a DataFrame:

##### Working with text data

Series and Index are equipped with a set of string processing methods that make it easy to operate on each element of the array. Perhaps most importantly, these methods exclude missing/NA values automatically. These are accessed via the `str` attribute and generally have names matching the equivalent built-in string methods. If you do want literal replacement of a string, you can set the optional `regex` parameter to False, rather than escaping each character. In this case both pat and repl must be strings. The `replace` method can also take a callable as replacement. It is called on every `pat` using `re.sub()`. The callable should expect one positional argument and return a string. The `replace` method also accepts a compiled regular expression object from `re.compile()` as a pattern. All flags should be included in the compiled regular expression object.
There are several ways to concatenate a Series or Index, either with itself or others, all based on `cat()`
By default, missing values are ignored. Using `na_rep`, they can be given a representation. The first argument to `cat()` can be a list-like object, provided that it matches the length of the calling Series. Missing values on either side will result in missing values in the result as well, unless `na_rep` is specified.
The parameter others can also be two-dimensional. In this case, the number or rows must match the lengths of the calling Series. For concatenation with a Series or DataFrame, it is possible to align the indexes before concatenation by setting the `join` keyword. The usual options are available for join (one of 'left', 'outer', 'inner', 'right'). In particular, alignment also means that the different lengths do not need to coincide anymore.
The same alignment can be used when others is a DataFrame. Several array-like items can be combined in a list-like container. All elements without an index within the passed list-like must match in length to the calling Series, but Series and Index may have arbitrary length (as long as alignment is not disabled with join=None). If using join='right' on a list-like of others that contains different indexes, the union of these indexes will be used as the basis for the final concatenation. You can use [] notation to directly index by position locations. If you index past the end of the string, the result will be a NaN.

The extract method accepts a regular expression with at least one capture group. Extracting a regular expression with more than one group returns a DataFrame with one column per group.
Elements that do not match return a row filled with NaN. Thus, a Series of messy strings can be “converted” into a like-indexed Series or DataFrame of cleaned-up or more useful strings, without necessitating `get()` to access tuples or `re.match` objects. The dtype of the result is always object, even if no match is found and the result only contains NaN. Note that any capture group names in the regular expression will be used for column names; otherwise capture group numbers will be used. Extracting a regular expression with one group returns a DataFrame with one column if `expand=True`.It returns a Series if `expand=False`. Calling on an Index with a regex with more than one capture group returns a DataFrame if `expand=True`. It raises ValueError if `expand=False`.

```python
pd.Series(['a1', 'b2', 'c3']).str.extract(r'(?P<letter>[ab])(?P<digit>\d)',expand=False)
```

the `extractall` method returns every match. The result of extractall is always a DataFrame with a MultiIndex on its rows. The last level of the MultiIndex is named match and indicates the order in the subject. When each subject string in the Series has exactly one match,then `extractall(pat).xs(0, level='match')` gives the same result as `extract(pat)`. Index also supports `.str.extractall`. It returns a DataFrame which has the same result as a `Series.str.extractall` with a default index. The distinction between match and contains is strictness: match relies on strict `re.match`, while contains relies on `re.search`. Methods like match, contains, startswith, and endswith take an extra `na` argument so missing values can be considered True or False:

| Method                 | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| `wrap()`               | Split long strings into lines with length less than a given width |
| `slice()`              | Slice each string in the Series                              |
| `slice_replace()`      | Replace slice in each string with passed value               |
| `findall()`            | Compute list of all occurrences of pattern/regex for each string |
| `len()`                |                                                              |
| `count()`              |                                                              |
| `partition/rpartition` |                                                              |

##### Working with missing data

As data comes in many shapes and forms, pandas aims to be flexible with regard to handling missing data. While NaN is the default missing value marker for reasons of computational speed and convenience, we need to be able to easily detect this value with data of different types: floating point, integer, boolean, and general object. 
To make detecting missing values easier, pandas provides the `isna()` and `notna()` functions, which are also methods on Series and DataFrame objects. One has to be mindful that in Python, the nan's don’t compare equal, but None's do. Note that pandas/NumPy uses the fact that np.nan != np.nan, and treats None like np.nan.
For `datetime64[ns]` types, NaT represents missing values. This is a pseudo-native sentinel value that can be represented by NumPy in a singular `dtype (datetime64[ns])`. pandas objects provide compatibility between NaT and NaN.
You can insert missing values by simply assigning to containers. The actual missing value used will be chosen based on the `dtype.numeric` containers will always use NaN regardless of the missing value type chosen; Likewise, datetime containers will always use NaT. For object containers, pandas will use the value given. Missing values propagate naturally through arithmetic operations between pandas objects.
You can also fillna using a dict or Series that is alignable. The labels of the dict or index of the Series must match the columns of the frame you wish to fill. Both Series and DataFrame objects have `interpolate()` that, by default, performs linear interpolation at missing data points.
The `method` argument gives access to fancier interpolation methods. If you have `scipy` installed, you can pass the name of a 1-d interpolation routine to method. The appropriate interpolation method will depend on the type of data you are working with. If you are dealing with a time series that is growing at an increasing rate, method='quadratic' may be appropriate. If you have values approximating a cumulative distribution function, then method='pchip' should work well. To fill missing values with goal of smooth plotting, consider method='akima'.

Often times we want to replace arbitrary values with other values. `replace()` in Series and `replace()` in DataFrame provides an efficient yet flexible way to perform such replacements. For a Series, you can replace a single value or a list of values by another value:

```python
ser = pd.Series([0., 1., 2., 3., 4.])
ser.replace(0, 5)
ser.replace([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
ser.replace({0: 10, 1: 100})
df = pd.DataFrame({'a': [0, 1, 2, 3, 4], 'b': [5, 6, 7, 8, 9]})
df.replace({'a': 0, 'b': 5}, 100)
```

All of the regular expression examples can also be passed with the `to_replace` argument as the `regex` argument. In this case the value argument must be passed explicitly by name or regex must be a nested dictionary. 

##### Nullable integer data type

Because NaN is a float, a column of integers with even one missing values is cast to floating-point dtype. Pandas provides a nullable integer array, which can be used by explicitly requesting the dtype:

```python
pd.Series([1, 2, np.nan, 4], dtype=pd.Int64Dtype())
```

Pandas can represent integer data with possibly missing values using arrays.IntegerArray. This is an extension types implemented within pandas. It is not the default dtype for integers, and will not be inferred; you must explicitly pass the dtype into array() or Series:

##### Categorical data

Categoricals are a pandas data type corresponding to categorical variables in statistics. A categorical variable takes on a limited, and usually fixed, number of possible values. In contrast to statistical categorical variables, categorical data might have an order, but numerical operations are not possible. All values of categorical data are either in categories or np.nan. Order is defined by the order of categories, not lexical order of the values. Internally, the data structure consists of a categories array and an integer array of codes which point to the real value in the categories array.

The categorical data type is useful in the following cases:

- A string variable consisting of only a few different values. Converting such a string variable to a categorical variable will save some memory.
- The lexical order of a variable is not the same as the logical order. By converting to a categorical and specifying an order on the categories, sorting and min/max will use the logical order instead of the lexical order.
- As a signal to other Python libraries that this column should be treated as a categorical variable.

###### Object creation

By specifying `dtype="category"` when constructing a Series. By converting an existing Series or column to a category dtype. By using special functions, such as cut(), which groups data into discrete bins. By passing a `pandas.Categorical` object to a Series or assigning it to a DataFrame.

```python
raw_cat = pd.Categorical(["a", "b", "c", "a"], categories=["b", "c", "d"],
                       ordered=False)
from pandas.api.types import CategoricalDtype
s = pd.Series(["a", "b", "c", "a"])
cat_type = CategoricalDtype(categories=["b", "c", "d"],ordered=True)
s_cat = s.astype(cat_type)
```

we passed `dtype='category'`, we used the default behavior: Categories are inferred from the data; Categories are unordered. To control those behaviors, instead of passing 'category', use an instance of CategoricalDtype.

A categorical’s type is fully described by: categories: a sequence of unique values and no missing values; ordered: a boolean. This information can be stored in a CategoricalDtype. The categories argument is optional, which implies that the actual categories should be inferred from whatever is present in the data when the pandas.Categorical is created.

Two instances of CategoricalDtype compare equal whenever they have the same categories and order. When comparing two unordered categoricals, the order of the categories is not considered. Categorical data has a `categories` and a `ordered` property, which list their possible values and whether the ordering matters or not. These properties are exposed as `s.cat.categories` and `s.cat.ordered`. If you don’t manually specify categories and ordering, they are inferred from the passed arguments.
Renaming categories is done by assigning new values to the `Series.cat.categories` property or by using the `rename_categories()` method. Appending categories can be done by using the `add_categories()` method. Removing categories can be done by using the `remove_categories()` method. Values which are removed are replaced by np.nan. If you want to do remove and add new categories in one step, or simply set the categories to a predefined scale, use `set_categories()`. 

If categorical data is ordered, then the order of the categories has a meaning and certain operations are possible. If the categorical is unordered, .min()/.max() will raise a TypeError.
You can set categorical data to be ordered by using `as_ordered()` or unordered by using `as_unordered()`. These will by default return a new object. Reordering the categories is possible via the `Categorical.reorder_categories()` and the `Categorical.set_categories()` methods. For `Categorical.reorder_categories()`, all old categories must be included in the new categories and no new categories are allowed. This will necessarily make the sort order the same as the categories order.

Comparing categorical data with other objects is possible in three cases:

- Comparing equality (== and !=) to a list-like objectof the same length as the categorical data.
- All comparisons (==, !=, >, >=, <, and <=) of categorical data to another categorical Series, when ordered==True and the categories are the same.
- All comparisons of a categorical data to a scalar.

All other comparisons, especially “non-equality” comparisons of two categoricals with different categories or a categorical with any list-like object, will raise a TypeError.
You can concat two DataFrames containing categorical data together, but the categories of these categoricals need to be the same. If you want to combine categoricals that do not necessarily have the same categories, the `union_categoricals()` function will combine a list-like of categoricals. The new categories will be the union of the categories being combined. By default, Series or DataFrame concatenation which contains the same categories results in category dtype, otherwise results in object dtype. Use `.astype` or `union_categoricals` to get category result. Missing values should not be included in the Categorical’s categories, only in the values. Instead, it is understood that NaN is different, and is always a possibility. In the Categorical’s codes, missing values will always have a code of -1.

##### Time series

pandas captures 4 general time related concepts: Date times: A specific date and time with timezone support; Time deltas: An absolute time duration; Time spans: A span of time defined by a point in time and its associated frequency; Date offsets: A relative time duration that respects calendar arithmetic.

| Concept      | Scalar Class | Array Class      | Data Type         | Creation Method                 |
| ------------ | ------------ | ---------------- | ----------------- | ------------------------------- |
| Date times   | `Timestamp`  | `DatetimeIndex`  | `datime64[ns]`    | `to_datetime,date_range`        |
| Time deltas  | `Timedelta`  | `TimedeltaIndex` | `timedelta64[ns]` | `to_timedelta, timedelta_range` |
| Time spans   | `Period`     | `PeriodIndex`    | `period[freq]`    | `Period, period_range`          |
| Date offsets | `DateOffset` | `None`           | `None`            | `DateOffset`                    |

Timestamped data is the most basic type of time series data that associates values with points in time. For pandas objects it means using the points in time.
However, in many cases it is more natural to associate things like change variables with a time span instead. The span represented by Period can be specified explicitly, or inferred from datetime string format.
pandas allows you to capture both representations and convert between them. Under the hood, pandas represents timestamps using instances of Timestamp and sequences of timestamps using instances of DatetimeIndex. For regular time spans, pandas uses Period objects for scalar values and PeriodIndex for sequences of spans. 