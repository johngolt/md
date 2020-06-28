#### Merge, join and concatenate

For binary operations on two ``Series`` or ``DataFrame`` objects, Pandas will align indices in the process of performing the operation.

A similar type of alignment takes place for *both* columns and indices when performing operations on ``DataFrame``s:

When performing operations between a ``DataFrame`` and a ``Series``, the index and column alignment is similarly maintained.
Operations between a ``DataFrame`` and a ``Series`` are similar to operations between a two-dimensional and one-dimensional NumPy array.

###### Concatenating objects

Pandas has a function, ``pd.concat()``, which has a similar syntax to ``np.concatenate`` but contains a number of options that we'll discuss momentarily:

```python
# Signature in Pandas v0.18
pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False,
          keys=None, levels=None, names=None, verify_integrity=False,
          copy=True)
```

``pd.concat()`` can be used for a simple concatenation of ``Series`` or ``DataFrame`` objects, just as ``np.concatenate()`` can be used for simple concatenations of arrays:

If you'd like to simply verify that the indices in the result of ``pd.concat()`` do not overlap, you can specify the ``verify_integrity`` flag.
With this set to True, the concatenation will raise an exception if there are duplicate indices.

Sometimes the index itself does not matter, and you would prefer it to simply be ignored.
This option can be specified using the ``ignore_index`` flag.
With this set to true, the concatenation will create a new integer index for the resulting ``Series``:

Another option is to use the ``keys`` option to specify a label for the data sources; the result will be a hierarchically indexed series containing the data:

By default, the entries for which no data is available are filled with NA values.
To change this, we can specify one of several options for the ``join`` and ``join_axes`` parameters of the concatenate function.
By default, the join is a union of the input columns (``join='outer'``), but we can change this to an intersection of the columns using ``join='inner'``:

Another option is to directly specify the index of the remaininig colums using the ``join_axes`` argument, which takes a list of index objects.
Here we'll specify that the returned columns should be the same as those of the first input:

Because direct array concatenation is so common, ``Series`` and ``DataFrame`` objects have an ``append`` method that can accomplish the same thing in fewer keystrokes.
For example, rather than calling ``pd.concat([df1, df2])``, you can simply call ``df1.append(df2)``:

Keep in mind that unlike the ``append()`` and ``extend()`` methods of Python lists, the ``append()`` method in Pandas does not modify the original object–instead it creates a new object with the combined data.
It also is not a very efficient method, because it involves creation of a new index *and* data buffer.

The `concat()` function does all of the heavy lifting of performing concatenation operations along an axis while performing optional set logic (union or intersection) of the indexes (if any) on the other axes. Note that “if any” because there is only a single possible axis of concatenation for Series.

```python
 result = pd.concat(frames, keys=['x', 'y', 'z'])
```

![](../picture/1/152.png)

When gluing together multiple `DataFrames`, you have a choice of how to handle the other axes. This can be done in the following two ways: Take the union of them all, `join='outer'`. Take the intersection, `join='inner'`.

```python
result = pd.concat([df1, df4], axis=1).reindex(df1.index)
```

![](../picture/1/153.png)

A useful shortcut to `concat()` are the `append()` instance methods on Series and `DataFrame`.  They concatenate along `axis=0`. Ignoring index on the concatenation axis, use the `ignore_index` argument. You can concatenate a mix of `Series` and `DataFrame` objects. The `Series` will be transformed to `DataFrame` with the column name as the name of the `Series`. If unnamed `Series` are passed they will be numbered consecutively. A fairly common use of the `keys` argument is to override the column names when creating a new `DataFrame` based on existing `Series`. Notice how the default behavior consists on letting the resulting `DataFrame` inherit the parent `Series`’ name, when these existed. 

```python
pieces = {'x': df1, 'y': df2, 'z': df3}
```

Through the `keys` argument we can override the existing column names. You can also pass a dict to `concat` in which case the dict keys will be used for the `keys` argument.



Perhaps the simplest type of merge expresion is the one-to-one join, which is in many ways very similar to the column-wise concatenation.
As a concrete example, consider the following two ``DataFrames`` which contain information on several employees in a company:

The ``pd.merge()`` function recognizes that each ``DataFrame`` has an "employee" column, and automatically joins using this column as a key.
The result of the merge is a new ``DataFrame`` that combines the information from the two inputs.
Notice that the order of entries in each column is not necessarily maintained: in this case, the order of the "employee" column differs between ``df1`` and ``df2``, and the ``pd.merge()`` function correctly accounts for this.
Additionally, keep in mind that the merge in general discards the index, except in the special case of merges by index.

Many-to-one joins are joins in which one of the two key columns contains duplicate entries.
For the many-to-one case, the resulting ``DataFrame`` will preserve those duplicate entries as appropriate.
The resulting ``DataFrame`` has an aditional column with the "supervisor" information, where the information is repeated in one or more locations as required by the inputs.

Many-to-many joins are a bit confusing conceptually, but are nevertheless well defined.
If the key column in both the left and right array contains duplicates, then the result is a many-to-many merge.
This will be perhaps most clear with a concrete example.
Consider the following, where we have a ``DataFrame`` showing one or more skills associated with a particular group.
By performing a many-to-many join, we can recover the skills associated with any individual person:

We've already seen the default behavior of ``pd.merge()``: it looks for one or more matching column names between the two inputs, and uses this as the key.
However, often the column names will not match so nicely, and ``pd.merge()`` provides a variety of options for handling this.

Most simply, you can explicitly specify the name of the key column using the ``on`` keyword, which takes a column name or a list of column names:

At times you may wish to merge two datasets with different column names; for example, we may have a dataset in which the employee name is labeled as "name" rather than "employee".
In this case, we can use the ``left_on`` and ``right_on`` keywords to specify the two column names:

Sometimes, rather than merging on a column, you would instead like to merge on an index.

If you'd like to mix indices and columns, you can combine ``left_index`` with ``right_on`` or ``left_on`` with ``right_index`` to get the desired behavior:

pandas provides a single function, `merge()`, as the entry point for all standard database join operations between DataFrame or named Series objects. If left is a DataFrame or named Series and right is a subclass of DataFrame, the return type will still be DataFrame. The related `join()` method, uses merge internally for the index-on-index and column(s)-on-index join. If you are joining on index only, you may wish to use `DataFrame.join` to save yourself some typing.
 There are several cases to consider which are very important to understand: one-to-one joins, many-to-one joins, many-to-many joins. In standard relational algebra, if a key combination appears more than once in both tables, the resulting table will have the Cartesian product of the associated data.
The `how` argument to `merge` specifies how to determine which keys are to be included in the resulting table. If a key combination does not appear in either the left or right tables, the values in the joined table will be NA.

| method | join Name        | Description                               |
| ------ | ---------------- | ----------------------------------------- |
| left   | left outer join  | use keys from left frame only             |
| right  | right outer join | use keys from right frame only            |
| outer  | full outer join  | use union of keys from both frames        |
| inner  | inner join       | use intersection of keys from both frames |

Users can use the `validate` argument to automatically check whether there are unexpected duplicates in their merge keys. Key uniqueness is checked before merge operations and so should protect against memory overflows. Checking key uniqueness is also a good way to ensure user data structures are as expected.

```python
result = pd.merge(left, right, on='B', how='outer', validate="one_to_one")
pd.merge(left, right, on='B', how='outer', validate="one_to_many")
```

You can join a singly-indexed DataFrame with a level of a MultiIndexed DataFrame. The level will match on the name of the index of the singly-indexed frame against a level name of the MultiIndexed frame. Strings passed as the `on, left_on`, and `right_on` parameters may refer to either column names or index level names. This enables merging DataFrame instances on a combination of index levels and columns without resetting indexes.

 The merge `suffixes` argument takes a tuple of list of strings to append to overlapping column names in the input DataFrames to disambiguate the result columns.



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

##### Group By

By “group by” we are referring to a process involving one or more of the following steps: Splitting the data into groups based on some criteria; Applying a function to each group independently; Combining the results into a data structure.
In the apply step, we might wish to do one of the following: Aggregation: compute a summary statistic for each group; Transformation: perform some group-specific computations and return a like-indexed object; Filtration: discard some groups, according to a group-wise computation that evaluates True or False. Some combination of the above

###### Splitting an object into groups

pandas objects can be split on any of their axes. The abstract definition of grouping is to provide a mapping of labels to group names.  The mapping can be specified many different ways:

- A Python function, to be called on each of the axis labels.
- A list or NumPy array of the same length as the selected axis.
- A dict or Series, providing a label -> group name mapping.
- For DataFrame objects, a string indicating a column to be used to group. 
- For DataFrame objects, a string indicating an index level to be used to group.
- A list of any of the above things.

pandas Index objects support duplicate values. If a non-unique index is used as the group key in a groupby operation, all values for the same index value will be considered to be in one group and thus the output of aggregation functions will only contain unique index values.
Note that no splitting occurs until it’s needed. Creating the `GroupBy` object only verifies that you’ve passed a valid mapping.

By default the group keys are sorted during the groupby operation. You may however pass `sort=False` for potential speedups. The `groups` attribute is a `dict` whose keys are the computed unique groups and corresponding values being the axis labels belonging to each group. 

With hierarchically-indexed data, it’s quite natural to group by one of the levels of the hierarchy. If the `MultiIndex` has names specified, these can be passed instead of the `level` number. Grouping with multiple levels is supported. A `DataFrame` may be grouped by a combination of columns and index levels by specifying the column names as strings and the index `levels` as `pd.Grouper` objects. 

```python
df.groupby([pd.Grouper(level=1), 'A']).sum()
df.groupby(['A', 'B']).get_group(('bar', 'one'))
```

Once you have created the `GroupBy` object from a `DataFrame`, you might want to do something different for each of the columns. Thus, using [] similar to getting a column from a `DataFrame`, you can do. With the `GroupBy` object in hand, iterating through the grouped data is very natural and functions similarly to `itertools.groupby()`. A single group can be selected using `get_group()`

Notice that what is returned is not a set of ``DataFrame``s, but a ``DataFrameGroupBy`` object.
This object is where the magic is: you can think of it as a special view of the ``DataFrame``, which is poised to dig into the groups but does no actual computation until the aggregation is applied.
This "lazy evaluation" approach means that common aggregates can be implemented very efficiently in a way that is almost transparent to the user.

To produce a result, we can apply an aggregate to this ``DataFrameGroupBy`` object, which will perform the appropriate apply/combine steps to produce the desired result:

The ``GroupBy`` object is a very flexible abstraction.
In many ways, you can simply treat it as if it's a collection of ``DataFrame``s, and it does the difficult things under the hood.

The ``GroupBy`` object supports column indexing in the same way as the ``DataFrame``, and returns a modified ``GroupBy`` object.


The ``GroupBy`` object supports direct iteration over the groups, returning each group as a ``Series`` or ``DataFrame``:

###### Aggregation

the ``aggregate()`` method allows for even more flexibility. It can take a string, a function, or a list thereof, and compute all the aggregates at once.

Once the GroupBy object has been created, several methods are available to perform a computation on the grouped data. An obvious one is aggregation via the `aggregate()` or equivalently `agg()` method. Any function which reduces a Series to a scalar value is an aggregation function and will work, `df.groupby('A').agg(lambda ser: 1)`. 

With grouped Series you can also pass a list or dict of functions to do aggregation with, outputting a DataFrame. On a grouped DataFrame, you can pass a list of functions to apply to each column, which produces an aggregated result with a hierarchical index. 

To support column-specific aggregation with control over the output column names, pandas accepts the special syntax in GroupBy.agg(), known as “named aggregation”, where The keywords are the output column names, The values are tuples whose first element is the column to select and the second element is the aggregation to apply to that column. Pandas provides the `pandas.NamedAgg` namedtuple with the fields `['column', 'aggfunc']` to make it clearer what the arguments are. Plain tuples are allowed as well.

```python
 animals.groupby("kind").agg(min_height=pd.NamedAgg(column='height', aggfunc='min'),max_height=pd.NamedAgg(column='height', aggfunc='max'),
  average_weight=pd.NamedAgg(column='weight', aggfunc=np.mean)
# pandas.NamedAgg is just a namedtuple. Plain tuples are allowed as well.
animals.groupby("kind").agg(min_height=('height', 'min'),max_height=('height', 'max'),average_weight=('weight', np.mean),
```

Additional keyword arguments are not passed through to the aggregation functions. Only pairs of (column, aggfunc) should be passed as. If your aggregation functions requires additional arguments, partially apply them with `functools.partial()`. By passing a dict to `aggregate` you can apply a different aggregation to the columns of a DataFrame.

###### Transformation

The transform method returns an object that is indexed the same as the one being grouped. The transform function must: 

- Return a result that is either the same size as the group chunk or broadcastable to the size of the group chunk. 
- Operate column-by-column on the group chunk. The transform is applied to the first group chunk using `chunk.apply`. 
- Not perform in-place operations on the group chunk. Group chunks should be treated as immutable, and changes to a group chunk may produce unexpected results. 
- operates on the entire group chunk. If this is supported, a fast path is used starting from the *second* chunk.

Transformation functions that have lower dimension outputs are broadcast to match the shape of the input array. it is possible to use `resample(), expanding()` and `rolling()` as methods on groupbys.

```python
df_re.groupby('A').rolling(4).B.mean()
dff.groupby('B').filter(lambda x: len(x) > 2)
```

###### Filtration

The `filter` method returns a subset of the original object. The argument of `filter` must be a function that, applied to the group as a whole, returns `True` or `False`. For `DataFrames` with multiple columns, filters should explicitly specify a column as the filter criterion.

For these, use the `apply` function, which can be substituted for both aggregate and transform in many standard use cases. However, apply can handle some exceptional use cases, apply on a Series can operate on a returned value from the applied function, that is itself a series, and possibly upcast the result to a DataFrame:

The ``apply()`` method lets you apply an arbitrary function to the group results.
The function should take a ``DataFrame``, and return either a Pandas object or a scalar; the combine operation will be tailored to the type of output returned.

##### Working with text data

Pandas includes features to address both this need for vectorized string operations and for correctly handling missing data via the ``str`` attribute of Pandas Series and Index objects containing strings.

Series and Index are equipped with a set of string processing methods that make it easy to operate on each element of the array. Perhaps most importantly, these methods exclude missing/NA values automatically. These are accessed via the `str` attribute and generally have names matching the equivalent built-in string methods.

If you do want literal replacement of a string, you can set the optional `regex` parameter to `False`, rather than escaping each character. In this case both `pat` and `repl` must be strings.

 ```python
dollars = pd.Series(['12', '-$10', '$10,000'])
dollars = pd.Series(['12', '-$10', '$10,000'])
dollars.str.replace('-$', '-', regex=False)
 ```

The `replace` method can also take a callable as replacement. It is called on every `pat` using `re.sub()`. The callable should expect one positional argument and return a string. 

```python
pat = r'[a-z]+'
def repl(m):
     return m.group(0)[::-1]
pd.Series(['foo 123', 'bar baz', np.nan]).str.replace(pat, repl)
pat = r"(?P<one>\w+) (?P<two>\w+) (?P<three>\w+)"
def repl(m):
     return m.group('two').swapcase()
pd.Series(['Foo Bar Baz', np.nan]).str.replace(pat, repl)
```

The `replace` method also accepts a compiled regular expression object from `re.compile()` as a pattern. All flags should be included in the compiled regular expression object.

```python
regex_pat = re.compile(r'^.a|dog', flags=re.IGNORECASE)
s3.str.replace(regex_pat, 'XX-XX ')
```

There are several ways to concatenate a Series or Index, either with itself or others, all based on `cat()`

```python
pd.Series(['a', 'b', 'c', 'd']).str.cat(sep=',')
s.str.cat(['A', 'B', 'C', 'D'])
v = pd.Series(['z', 'a', 'b', 'd', 'e'], index=[-1, 0, 1, 3, 4])
s.str.cat(v, join='left', na_rep='-')
```

By default, missing values are ignored. Using `na_rep`, they can be given a representation. The first argument to `cat()` can be a list-like object, provided that it matches the length of the calling Series. Missing values on either side will result in missing values in the result as well, unless `na_rep` is specified.
The parameter `others` can also be two-dimensional. In this case, the number or rows must match the lengths of the calling Series. For concatenation with a Series or DataFrame, it is possible to align the indexes before concatenation by setting the `join` keyword. The usual options are available for join (one of 'left', 'outer', 'inner', 'right'). In particular, alignment also means that the different lengths do not need to coincide anymore.
The same alignment can be used when `others` is a DataFrame. Several array-like items can be combined in a list-like container. All elements without an index within the passed list-like must match in length to the calling Series, but Series and Index may have arbitrary length (as long as alignment is not disabled with join=None). If using join='right' on a list-like of `others` that contains different indexes, the union of these indexes will be used as the basis for the final concatenation. 

You can use `[]` notation to directly index by position locations. If you index past the end of the string, the result will be a `NaN`.

###### Extracting substrings

The `extract` method accepts a regular expression with at least one capture group. Extracting a regular expression with more than one group returns a DataFrame with one column per group.

```python
pd.Series(['a1', 'b2', 'c3']).str.extract(r'([ab])(\d)', expand=False)
```

Elements that do not match return a row filled with NaN. Thus, a Series of messy strings can be “converted” into a like-indexed Series or DataFrame of cleaned-up or more useful strings, without necessitating `get()` to access tuples or `re.match` objects. The `dtype` of the result is always object, even if no match is found and the result only contains NaN. Note that any capture group names in the regular expression will be used for column names; otherwise capture group numbers will be used. 

Extracting a regular expression with one group returns a DataFrame with one column if `expand=True`.It returns a Series if `expand=False`. Calling on an Index with a regex with more than one capture group returns a DataFrame if `expand=True`. It raises ValueError if `expand=False`.

```python
pd.Series(['a1', 'b2', 'c3']).str.extract(r'(?P<letter>[ab])(?P<digit>\d)',expand=False)
```

the `extractall` method returns every match. The result of `extractall` is always a DataFrame with a `MultiIndex` on its rows. The last level of the `MultiIndex` is named match and indicates the order in the subject. When each subject string in the Series has exactly one match,then `extractall(pat).xs(0, level='match')` gives the same result as `extract(pat)`. Index also supports `.str.extractall`. It returns a DataFrame which has the same result as a `Series.str.extractall` with a default index. 

The distinction between `match` and `contains` is strictness: `match` relies on strict `re.match`, while `contains` relies on `re.search`. Methods like `match`, `contains, startswith`, and `endswith` take an extra `na` argument so missing values can be considered `True` or `False`.

| Method            | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `wrap()`          | Split long strings into lines with length less than a given width |
| `slice()`         | Slice each string in the Series                              |
| `slice_replace()` | Replace slice in each string with passed value               |
| `findall()`       | Compute list of all occurrences of pattern/regex for each string |
| `count()`         | Count occurrences of pattern                                 |

Nearly all Python's built-in string methods are mirrored by a Pandas vectorized string method. Here is a list of Pandas ``str`` methods that mirror Python string methods:

|              |                  |                  |                  |
| ------------ | ---------------- | ---------------- | ---------------- |
| ``len()``    | ``lower()``      | ``translate()``  | ``islower()``    |
| ``ljust()``  | ``upper()``      | ``startswith()`` | ``isupper()``    |
| ``rjust()``  | ``find()``       | ``endswith()``   | ``isnumeric()``  |
| ``center()`` | ``rfind()``      | ``isalnum()``    | ``isdecimal()``  |
| ``zfill()``  | ``index()``      | ``isalpha()``    | ``split()``      |
| ``strip()``  | ``rindex()``     | ``isdigit()``    | ``rsplit()``     |
| ``rstrip()`` | ``capitalize()`` | ``isspace()``    | ``partition()``  |
| ``lstrip()`` | ``swapcase()``   | ``istitle()``    | ``rpartition()`` |

In addition, there are several methods that accept regular expressions to examine the content of each string element, and follow some of the API conventions of Python's built-in ``re`` module:

| Method         | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| ``match()``    | Call ``re.match()`` on each element, returning a boolean.    |
| ``extract()``  | Call ``re.match()`` on each element, returning matched groups as strings. |
| ``findall()``  | Call ``re.findall()`` on each element                        |
| ``replace()``  | Replace occurrences of pattern with some other string        |
| ``contains()`` | Call ``re.search()`` on each element, returning a boolean    |
| ``count()``    | Count occurrences of pattern                                 |
| ``split()``    | Equivalent to ``str.split()``, but accepts regexps           |
| ``rsplit()``   | Equivalent to ``str.rsplit()``, but accepts regexps          |

Finally, there are some miscellaneous methods that enable other convenient operations:

| Method              | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| ``get()``           | Index each element                                           |
| ``slice()``         | Slice each element                                           |
| ``slice_replace()`` | Replace slice in each element with passed value              |
| ``cat()``           | Concatenate strings                                          |
| ``repeat()``        | Repeat values                                                |
| ``normalize()``     | Return Unicode form of string                                |
| ``pad()``           | Add whitespace to left, right, or both sides of strings      |
| ``wrap()``          | Split long strings into lines with length less than a given width |
| ``join()``          | Join strings in each element of the Series with passed separator |
| ``get_dummies()``   | extract dummy variables as a dataframe                       |

##### Working with missing data

The first sentinel value used by Pandas is ``None``, a Python singleton object that is often used for missing data in Python code.
Because it is a Python object, ``None`` cannot be used in any arbitrary NumPy/Pandas array, but only in arrays with data type ``'object'`` 

This ``dtype=object`` means that the best common type representation NumPy could infer for the contents of the array is that they are Python objects.
While this kind of object array is useful for some purposes, any operations on the data will be done at the Python level, with much more overhead than the typically fast operations seen for arrays with native types:

The use of Python objects in an array also means that if you perform aggregations like ``sum()`` or ``min()`` across an array with a ``None`` value, you will generally get an error:


The other missing data representation, ``NaN``, is different; it is a special floating-point value recognized by all systems that use the standard IEEE floating-point representation:

Notice that NumPy chose a native floating-point type for this array: this means that unlike the object array from before, this array supports fast operations pushed into compiled code.
You should be aware that ``NaN`` is a bit like a data virus–it infects any other object it touches.
Regardless of the operation, the result of arithmetic with ``NaN`` will be another ``NaN``:
NumPy does provide some special aggregations that will ignore these missing values:
`np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2)`

Keep in mind that ``NaN`` is specifically a floating-point value; there is no equivalent NaN value for integers, strings, or other types.

For types that don't have an available sentinel value, Pandas automatically type-casts when NA values are present.
For example, if we set a value in an integer array to ``np.nan``, it will automatically be upcast to a floating-point type to accommodate the NA:

The following table lists the upcasting conventions in Pandas when NA values are introduced:

| Typeclass    | Conversion When Storing NAs | NA Sentinel Value      |
| ------------ | --------------------------- | ---------------------- |
| ``floating`` | No change                   | ``np.nan``             |
| ``object``   | No change                   | ``None`` or ``np.nan`` |
| ``integer``  | Cast to ``float64``         | ``np.nan``             |
| ``boolean``  | Cast to ``object``          | ``None`` or ``np.nan`` |

Keep in mind that in Pandas, string data is always stored with an ``object`` dtype.

there are several useful methods for detecting, removing, and replacing null values in Pandas data structures.
They are:

- ``isnull()``: Generate a boolean mask indicating missing values
- ``notnull()``: Opposite of ``isnull()``
- ``dropna()``: Return a filtered version of the data
- ``fillna()``: Return a copy of the data with missing values filled or imputed

We cannot drop single values from a ``DataFrame``; we can only drop full rows or full columns.
Depending on the application, you might want one or the other, so ``dropna()`` gives a number of options for a ``DataFrame``.

By default, ``dropna()`` will drop all rows in which *any* null value is present:

Alternatively, you can drop NA values along a different axis; ``axis=1`` drops all columns containing a null value:

But this drops some good data as well; you might rather be interested in dropping rows or columns with *all* NA values, or a majority of NA values.
This can be specified through the ``how`` or ``thresh`` parameters, which allow fine control of the number of nulls to allow through.

The default is ``how='any'``, such that any row or column (depending on the ``axis`` keyword) containing a null value will be dropped.
You can also specify ``how='all'``, which will only drop rows/columns that are *all* null values:

 While `NaN` is the default missing value marker for reasons of computational speed and convenience, we need to be able to easily detect this value with data of different types. To make detecting missing values easier, pandas provides the `isna()` and `notna()` functions. One has to be mindful that in Python, the nan's don’t compare equal, but None's do. Note that pandas/NumPy uses the fact that `np.nan != np.nan`, and treats `None` like `np.nan`.

Because `NaN` is a float, a column of integers with even one missing values is cast to floating-point `dtype`. The actual missing value used will be chosen based on the `dtype.numeric` containers will always use `NaN` regardless of the missing value type chosen; Likewise, datetime containers will always use `NaT`. For object containers, pandas will use the value given. Missing values propagate naturally through arithmetic operations between pandas objects.

###### filling missing values

```python
df2.fillna(0)#Replace NA with a scalar value
df.fillna(method='pad')#Fill gaps forward or backward
df.fillna(method='pad', limit=1) #limit the amount of filling
dff.where(pd.notna(dff), dff.mean(), axis='columns')
```

You can also `fillna` using a dict or Series that is alignable. The labels of the dict or index of the Series must match the columns of the frame you wish to fill. 

Both Series and DataFrame objects have `interpolate()` that, by default, performs linear interpolation at missing data points. The `method` argument gives access to fancier interpolation methods. If you have `scipy` installed, you can pass the name of a 1-d interpolation routine to method. The appropriate interpolation method will depend on the type of data you are working with. If you are dealing with a time series that is growing at an increasing rate, `method='quadratic'` may be appropriate. If you have values approximating a cumulative distribution function, then `method='pchip'` should work well. To fill missing values with goal of smooth plotting, consider `method='akima'`.

###### Replacing generic values

`replace()` in Series and `replace()` in DataFrame provides an efficient yet flexible way to perform such replacements. For a Series, you can replace a single value or a list of values by another value.

```python
ser = pd.Series([0., 1., 2., 3., 4.])
ser.replace(0, 5)
ser.replace([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
ser.replace({0: 10, 1: 100})
#For a DataFrame, you can specify individual values by column
df = pd.DataFrame({'a': [0, 1, 2, 3, 4], 'b': [5, 6, 7, 8, 9]})
df.replace({'a': 0, 'b': 5}, 100)
```

All of the regular expression examples can also be passed with the `to_replace` argument as the `regex` argument. In this case the value argument must be passed explicitly by name or regex must be a nested dictionary. 

```python
d = {'a': list(range(4)), 'b': list('ab..'), 'c': ['a', 'b', np.nan, 'd']}
df = pd.DataFrame(d)
df.replace('.', np.nan) # str->str
df.replace(r'\s*\.\s*', np.nan, regex=True) # regex->regex
df.replace(['a', '.'], ['b', np.nan]) # list->list
df.replace([r'\.', r'(a)'], ['dot', r'\1stuff'], regex=True) #list of regex -> list of regex
df.replace({'b': '.'}, {'b': np.nan}) # dict->dict
df.replace({'b': r'\s*\.\s*'}, {'b': np.nan}, regex=True)#dict of regex-> dict

```

##### Categorical data

`Categoricals` are a pandas data type corresponding to categorical variables in statistics. A categorical variable takes on a limited, and usually fixed, number of possible values. In contrast to statistical categorical variables, categorical data might have an order, but numerical operations are not possible. All values of categorical data are either in categories or np.nan. Order is defined by the order of categories, not lexical order of the values. Internally, the data structure consists of a categories array and an integer array of codes which point to the real value in the categories array.

The categorical data type is useful in the following cases:

- A string variable consisting of only a few different values. Converting such a string variable to a categorical variable will save some memory.
- The lexical order of a variable is not the same as the logical order. By converting to a categorical and specifying an order on the categories, sorting and min/max will use the logical order instead of the lexical order.
- As a signal to other Python libraries that this column should be treated as a categorical variable.

###### Object creation

By specifying `dtype="category"` when constructing a Series. By converting an existing Series or column to a category dtype. By using special functions, such as `cut()`, which groups data into discrete bins.

```python
raw_cat = pd.Categorical(["a", "b", "c", "a"], categories=["b", "c", "d"],
                       ordered=False)
from pandas.api.types import CategoricalDtype
s = pd.Series(["a", "b", "c", "a"])
cat_type = CategoricalDtype(categories=["b", "c", "d"],ordered=True)
s_cat = s.astype(cat_type)
```

we passed `dtype='category'`, we used the default behavior: Categories are inferred from the data; Categories are unordered. To control those behaviors, instead of passing 'category', use an instance of `CategoricalDtype`.

A categorical’s type有两部分：categories: a sequence of unique values and no missing values; ordered: a boolean. This information can be stored in a `CategoricalDtype`. The categories argument is optional, which implies that the actual categories should be inferred from whatever is present in the data when the `pandas.Categorical` is created. These properties are exposed as `s.cat.categories` and `s.cat.ordered`.

Two instances of CategoricalDtype compare equal whenever they have the same categories and order. When comparing two unordered categoricals, the order of the categories is not considered. All instances of `CategoricalDtype` compare equal to the string `'category'`.

Renaming categories is done by assigning new values to the `Series.cat.categories` property or by using the `rename_categories()` method. Appending categories can be done by using the `add_categories()` method. Removing categories can be done by using the `remove_categories()` method. Values which are removed are replaced by `np.nan`. If you want to do remove and add new categories in one step, or simply set the categories to a predefined scale, use `set_categories()`. 

If categorical data is ordered, then the order of the categories has a meaning and certain operations are possible. If the categorical is unordered, .min()/.max() will raise a `TypeError`.
You can set categorical data to be ordered by using `as_ordered()` or unordered by using `as_unordered()`. These will by default return a new object. Reordering the categories is possible via the `Categorical.reorder_categories()` and the `Categorical.set_categories()` methods. For `Categorical.reorder_categories()`, all old categories must be included in the new categories and no new categories are allowed. This will necessarily make the sort order the same as the categories order.

Comparing categorical data with other objects is possible in three cases:

- Comparing equality (== and !=) to a list-like object of the same length as the categorical data.
- All comparisons (==, !=, >, >=, <, and <=) of categorical data to another categorical Series, when `ordered==True` and the categories are the same.
- All comparisons of a categorical data to a scalar.

You can `concat` two DataFrames containing categorical data together, but the categories of these categoricals need to be the same. If you want to combine categoricals that do not necessarily have the same categories, the `union_categoricals()` function will combine a list-like of categoricals. The new categories will be the union of the categories being combined. 

By default, Series or DataFrame concatenation which contains the same categories results in category dtype, otherwise results in object dtype. Use `.astype` or `union_categoricals` to get category result. Missing values should not be included in the Categorical’s categories, only in the values. Instead, it is understood that NaN is different, and is always a possibility. In the Categorical’s codes, missing values will always have a code of -1.

##### Time series

pandas captures 4 general time related concepts: Date times: A specific date and time with timezone support; Time deltas: An absolute time duration; Time spans: A span of time defined by a point in time and its associated frequency; Date offsets: A relative time duration that respects calendar arithmetic.

| Concept      | Scalar Class | Array Class      | Data Type         | Creation Method                 |
| ------------ | ------------ | ---------------- | ----------------- | ------------------------------- |
| Date times   | `Timestamp`  | `DatetimeIndex`  | `datime64[ns]`    | `to_datetime,date_range`        |
| Time deltas  | `Timedelta`  | `TimedeltaIndex` | `timedelta64[ns]` | `to_timedelta, timedelta_range` |
| Time spans   | `Period`     | `PeriodIndex`    | `period[freq]`    | `Period, period_range`          |
| Date offsets | `DateOffset` | `None`           | `None`            | `DateOffset`                    |

The power of ``datetime`` and ``dateutil`` lie in their flexibility and easy syntax: you can use these objects and their built-in methods to easily perform nearly any operation you might be interested in.
Where they break down is when you wish to work with large arrays of dates and times:
just as lists of Python numerical variables are suboptimal compared to NumPy-style typed numerical arrays, lists of Python datetime objects are suboptimal compared to typed arrays of encoded dates.

The weaknesses of Python's datetime format inspired the NumPy team to add a set of native time series data type to NumPy.
The ``datetime64`` dtype encodes dates as 64-bit integers, and thus allows arrays of dates to be represented very compactly.
The ``datetime64`` requires a very specific input format:

One detail of the ``datetime64`` and ``timedelta64`` objects is that they are built on a *fundamental time unit*.
Because the ``datetime64`` object is limited to 64-bit precision, the range of encodable times is $2^{64}$ times this fundamental unit.
In other words, ``datetime64`` imposes a trade-off between *time resolution* and *maximum time span*.

Notice that the time zone is automatically set to the local time on the computer executing the code.
You can force any desired fundamental unit using one of many format codes;

The following table,lists the available format codes
along with the relative and absolute timespans that they can encode:

| Code   | Meaning     | Time span (relative) | Time span (absolute)   |
| ------ | ----------- | -------------------- | ---------------------- |
| ``Y``  | Year        | ± 9.2e18 years       | [9.2e18 BC, 9.2e18 AD] |
| ``M``  | Month       | ± 7.6e17 years       | [7.6e17 BC, 7.6e17 AD] |
| ``W``  | Week        | ± 1.7e17 years       | [1.7e17 BC, 1.7e17 AD] |
| ``D``  | Day         | ± 2.5e16 years       | [2.5e16 BC, 2.5e16 AD] |
| ``h``  | Hour        | ± 1.0e15 years       | [1.0e15 BC, 1.0e15 AD] |
| ``m``  | Minute      | ± 1.7e13 years       | [1.7e13 BC, 1.7e13 AD] |
| ``s``  | Second      | ± 2.9e12 years       | [ 2.9e9 BC, 2.9e9 AD]  |
| ``ms`` | Millisecond | ± 2.9e9 years        | [ 2.9e6 BC, 2.9e6 AD]  |
| ``us`` | Microsecond | ± 2.9e6 years        | [290301 BC, 294241 AD] |
| ``ns`` | Nanosecond  | ± 292 years          | [ 1678 AD, 2262 AD]    |
| ``ps`` | Picosecond  | ± 106 days           | [ 1969 AD, 1970 AD]    |
| ``fs`` | Femtosecond | ± 2.6 hours          | [ 1969 AD, 1970 AD]    |
| ``as`` | Attosecond  | ± 9.2 seconds        | [ 1969 AD, 1970 AD]    |

Pandas builds upon all the tools just discussed to provide a ``Timestamp`` object, which combines the ease-of-use of ``datetime`` and ``dateutil`` with the efficient storage and vectorized interface of ``numpy.datetime64``.
From a group of these ``Timestamp`` objects, Pandas can construct a ``DatetimeIndex`` that can be used to index data in a ``Series`` or ``DataFrame``; 

any of the ``Series`` indexing patterns we discussed in previous sections, passing values that can be coerced into dates:

the fundamental Pandas data structures for working with time series data:

- For *time stamps*, Pandas provides the ``Timestamp`` type. As mentioned before, it is essentially a replacement for Python's native ``datetime``, but is based on the more efficient ``numpy.datetime64`` data type. The associated Index structure is ``DatetimeIndex``.
- For *time Periods*, Pandas provides the ``Period`` type. This encodes a fixed-frequency interval based on ``numpy.datetime64``. The associated index structure is ``PeriodIndex``.
- For *time deltas* or *durations*, Pandas provides the ``Timedelta`` type. ``Timedelta`` is a more efficient replacement for Python's native ``datetime.timedelta`` type, and is based on ``numpy.timedelta64``. The associated index structure is ``TimedeltaIndex``.

Any ``DatetimeIndex`` can be converted to a ``PeriodIndex`` with the ``to_period()`` function with the addition of a frequency code; 

A ``TimedeltaIndex`` is created,when a date is subtracted from another:

To make the creation of regular date sequences more convenient, Pandas offers a few functions for this purpose: ``pd.date_range()`` for timestamps, ``pd.period_range()`` for periods, and ``pd.timedelta_range()`` for time deltas.
We've seen that Python's ``range()`` and NumPy's ``np.arange()`` turn a startpoint, endpoint, and optional stepsize into a sequence.
Similarly, ``pd.date_range()`` accepts a start date, an end date, and an optional frequency code to create a regular sequence of dates.
By default, the frequency is one day:

Fundamental to these Pandas time series tools is the concept of a frequency or date offset.
The following table summarizes the main codes available:

| Code  | Description  | Code   | Description          |
| ----- | ------------ | ------ | -------------------- |
| ``D`` | Calendar day | ``B``  | Business day         |
| ``W`` | Weekly       |        |                      |
| ``M`` | Month end    | ``BM`` | Business month end   |
| ``Q`` | Quarter end  | ``BQ`` | Business quarter end |
| ``A`` | Year end     | ``BA`` | Business year end    |
| ``H`` | Hours        | ``BH`` | Business hours       |
| ``T`` | Minutes      |        |                      |
| ``S`` | Seconds      |        |                      |
| ``L`` | Milliseonds  |        |                      |
| ``U`` | Microseconds |        |                      |
| ``N`` | nanoseconds  |        |                      |

The monthly, quarterly, and annual frequencies are all marked at the end of the specified period.
By adding an ``S`` suffix to any of these, they instead will be marked at the beginning:

| Code   | Description   |      | Code    | Description            |
| ------ | ------------- | ---- | ------- | ---------------------- |
| ``MS`` | Month start   |      | ``BMS`` | Business month start   |
| ``QS`` | Quarter start |      | ``BQS`` | Business quarter start |
| ``AS`` | Year start    |      | ``BAS`` | Business year start    |

Additionally, you can change the month used to mark any quarterly or annual code by adding a three-letter month code as a suffix:

- ``Q-JAN``, ``BQ-FEB``, ``QS-MAR``, ``BQS-APR``, etc.
- ``A-JAN``, ``BA-FEB``, ``AS-MAR``, ``BAS-APR``, etc.

In the same way, the split-point of the weekly frequency can be modified by adding a three-letter weekday code:

- ``W-SUN``, ``W-MON``, ``W-TUE``, ``W-WED``, etc.

On top of this, codes can be combined with numbers to specify other frequencies.
For example, for a frequency of 2 hours 30 minutes, we can combine the hour (``H``) and minute (``T``) codes as follows:

All of these short codes refer to specific instances of Pandas time series offsets, which can be found in the ``pd.tseries.offsets`` module.

One common need for time series data is resampling at a higher or lower frequency.
This can be done using the ``resample()`` method, or the much simpler ``asfreq()`` method.
The primary difference between the two is that ``resample()`` is fundamentally a *data aggregation*, while ``asfreq()`` is fundamentally a *data selection*.

For up-sampling, ``resample()`` and ``asfreq()`` are largely equivalent, though resample has many more options available.
In this case, the default for both methods is to leave the up-sampled points empty, that is, filled with NA values.
Just as with the ``pd.fillna()`` function discussed previously, ``asfreq()`` accepts a ``method`` argument to specify how values are imputed.

Another common time series-specific operation is shifting of data in time.
Pandas has two closely related methods for computing this: ``shift()`` and ``tshift()``
In short, the difference between them is that ``shift()`` *shifts the data*, while ``tshift()`` *shifts the index*.
In both cases, the shift is specified in multiples of the frequency.

##### Performance

Because NumPy evaluates each subexpression, In other words, every intermediate step is explicitly allocated in memory. If the ``x`` and ``y`` arrays are very large, this can lead to significant memory and computational overhead.
The `Numexpr` library gives you the ability to compute this type of compound expression element by element, without the need to allocate full intermediate arrays.

The ``eval()`` function in Pandas uses string expressions to efficiently compute operations using ``DataFrame``s. ``pd.eval()`` supports the ``&`` and ``|`` bitwise operators. In addition, it supports the use of the literal ``and`` and ``or`` in Boolean expressions. ``pd.eval()`` supports access to object attributes via the ``obj.attr`` syntax, and indexes via the ``obj[index]`` syntax. Just as Pandas has a top-level ``pd.eval()`` function, ``DataFrame``s have an ``eval()`` method that works in similar ways.
The benefit of the ``eval()`` method is that columns can be referred to *by name*.

In addition to the options just discussed, ``DataFrame.eval()``  also allows assignment to any column.
Let's use the ``DataFrame`` from before, which has columns ``'A'``, ``'B'``, and ``'C'``:

We can use ``df.eval()`` to create a new column ``'D'`` and assign to it a value computed from the other columns. In the same way, any existing column can be modified:

The ``@`` character here marks a *variable name* rather than a *column name*, and lets you efficiently evaluate expressions involving the two "namespaces": the namespace of columns, and the namespace of Python objects.
Notice that this ``@`` character is only supported by the ``DataFrame.eval()`` *method*, not by the ``pandas.eval()`` *function*, because the ``pandas.eval()`` function only has access to the one (Python) namespace.