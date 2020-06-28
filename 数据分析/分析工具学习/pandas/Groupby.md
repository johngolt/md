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