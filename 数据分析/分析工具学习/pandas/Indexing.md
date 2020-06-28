a ``Series`` object acts in many ways like a one-dimensional `Numpy` array, and in many ways like a standard Python dictionary. Like a dictionary, the ``Series`` object provides a mapping from a collection of keys to a collection of values. We can also use dictionary-like Python expressions and methods to examine the keys/indices and values. ``Series`` objects can even be modified with a dictionary-like syntax. Just as you can extend a dictionary by assigning to a new key, you can extend a ``Series`` by assigning to a new index value. A ``Series`` builds on this dictionary-like interface and provides array-style item selection via the same basic mechanisms as NumPy arrays – that is, *slices*, *masking*, and *fancy indexing*.

Recall that a ``DataFrame`` acts in many ways like a two-dimensional or structured array, and in other ways like a dictionary of ``Series`` structures sharing the same index. The first analogy we will consider is the ``DataFrame`` as a dictionary of related ``Series`` objects. The individual ``Series`` that make up the columns of the ``DataFrame`` can be accessed via dictionary-style indexing of the column name. Like with the ``Series`` objects discussed earlier, this dictionary-style syntax can also be used to modify the object, in this case adding a new column. Because Pandas is designed to work with NumPy, any NumPy ufunc will work on Pandas ``Series`` and ``DataFrame`` objects.

#### Indexing and selecting data

Object selection has had a number of user-requested additions in order to support more explicit location based indexing. Pandas now supports three types of multi-axis indexing.

- `.loc` is primarily label based, but may also be used with a boolean array. `.loc` will raise `KeyError` when the items are not found. Allowed inputs are: 
  - A single label;
  - A list or array of labels
  -  A slice object with labels 'a':'f', Note that contrary to usual python slices, both the start and the stop are included, when present in the index
  - A boolean array
  -  A callable function with one argument and that returns valid output for indexing，`iloc`中使用的函数，传入参数就是前面的`df`。
- `.iloc` is primarily integer position based, but may also be used with a boolean array. `.iloc` will raise `IndexError` if a requested indexer is out-of-bounds, except slice indexers which allow out-of-bounds indexing. Allowed inputs are: 
  - An integer
  -  A list or array of integers
  -  A slice object with ints
  - A boolean array
  -  A callable function with one argument and that returns valid output for indexing，`iloc`中使用的函数，传入参数就是前面的`df`.

`.loc`, `.iloc`, and also `[]` indexing can accept a `callable` as indexer. 

| Object Type | Selection        | Return Value Type                |
| ----------- | ---------------- | -------------------------------- |
| `Series`    | `series[label]`  | scalar value                     |
| `DataFrame` | `frame[colname]` | `Series` corresponding to column |

With Series, the syntax works exactly as with an `ndarray`, returning a slice of the values and the corresponding labels. With `DataFrame`, slicing inside of `[]` slices the rows. This is provided largely as a convenience since it is such a common operation.

###### Selection by label

`.loc` is strict when you present slicers that are not compatible (or convertible) with the index type. These will raise a `TypeError`. When using `.loc` with slices, if both the start and the stop labels are present in the index, then elements located between the two (including them) are returned. If at least one of the two is absent, but the index is sorted, and can be compared against start and stop labels, then slicing will still work as expected, by selecting labels which rank between the two. However, if at least one of the two is absent *and* the index is not sorted, an error will be raised

```python
s = pd.Series(list('abcde'), index=[0, 3, 2, 5, 4])
s.sort_index().loc[1:6]
```

`.loc`, `.iloc`, and also `[]` indexing can accept a `callable` as indexer. The `callable` must be a function with one argument (the calling Series or `DataFrame`) that returns valid output for indexing.

```python
bb.groupby(['year', 'team']).sum().loc[lambda df: df.r > 100]
```

###### $\text{Reindexing}$

if you want to select only *valid* keys, the following is idiomatic and efficient; it is guaranteed to preserve the `dtype` of the selection.

```python
s.loc[s.index.intersection([1,2,3])]  # select only valid keys
s.loc[s.index.intersection(labels)].reindex(labels)
#Having a duplicated index will raise for a `.reindex()`,会报错。
```

###### Selecting random samples

A random selection of rows or columns from a Series or `DataFrame` with the `sample()` method. By default, sample will return each row at most once, but one can also sample with replacement using the `replace` option. By default, each row has an equal probability of being selected, but you can pass the `sample` function sampling weights as `weights`. Missing values will be treated as a weight of zero, and `inf` values are not allowed. When applied to a `DataFrame`, you can use a column of the `DataFrame` as sampling weights by simply passing the name of the column as a string. sample also allows users to sample columns instead of rows using the `axis` argument. Finally, one can also set a seed for sample’s random number generator using the `random_state` argument. 

###### Boolean indexing

If you only want to access a scalar value, the fastest way is to use the `at` and `iat` methods, which are implemented on all of the data structures. Similarly to `loc`, at provides label based scalar lookups, while, `iat` provides integer based lookups analogously to `iloc`.

Another common operation is the use of boolean vectors to filter the data. The operators are: `|` for `or`, `&` for `and`, and `~` for `not`. These must be grouped by using parentheses, since by default Python will evaluate an expression such as `df.A > 2 & df.B < 3` as `df.A > (2 & df.B) < 3`. Using a boolean vector to index a Series works exactly as in a `NumPy`. You may select rows from a `DataFrame` using a boolean vector the same length as the `DataFrame’s` index. List comprehensions and the `map` method of Series can also be used to produce more complex criteria

###### Indexing with `isin`

Consider the `isin()` method of Series, which returns a boolean vector that is true wherever the Series elements exist in the passed list. This allows you to select rows where one or more columns have values you want. `DataFrame` also has an `isin()` method. When calling `isin`, pass a set of values as either an `array` or `dict`. If values is an `array`, `isin` returns a `DataFrame` of booleans that is the same shape as the original `DataFrame`, with True wherever the element is in the sequence of values. Oftentimes you’ll want to match certain values with certain columns. Just make values a `dict` where the key is the column, and the value is a list of items you want to check for. Combine `DataFrame’s` `isin` with the `any()` and `all()` methods to quickly select subsets of your data that meet a given criteria. 

```python
s_mi = pd.Series(np.arange(6),index=pd.MultiIndex.from_product([[0, 1], ['a', 'b', 'c']]))
s_mi.iloc[s_mi.index.isin([(1, 'a'), (2, 'b'), (0, 'c')])]
s_mi.iloc[s_mi.index.isin(['a', 'c', 'e'], level=1)]
```

###### The `where()` Method and Masking

Selecting values from a Series with a boolean vector generally returns a subset of the data. To guarantee that selection output has the same shape as the original data, you can use the `where` method in Series and `DataFrame`. In addition, `where` takes an optional `other` argument for replacement of values where the condition is `False`, in the returned copy. By default, `where` returns a modified copy of the data. There is an optional parameter `inplace` so that the original data can be modified without creating a copy. `Where` can also accept `axis` and `level` parameters to align the input when performing the `where`. `Where` can accept a callable as condition and other arguments. The function must be with one argument (the calling Series or `DataFrame`) and that returns valid output as condition and other argument. `mask()` is the inverse boolean operation of `where`.

```python
df2.where(df2 > 0, df2['A'], axis='index')# align the input
df3.where(lambda x: x > 4, lambda x: x + 10) # callable
```

###### The `query()` Method

```python
df = pd.DataFrame(np.random.rand(n, 3), columns=list('abc'))
df.query('(a < b) & (b < c)')
# If instead you don’t want to or cannot name your index, you can use the name index in your query expression.
df.query('index < b < c')
df.query('[1, 2] in c'); df.query('ilevel_0=="red"')
df.query('b == ["a", "b", "c"]')
df.query('~bools')#negate boolean expressions with not or the ~.
```

If the name of your index overlaps with a column name, the column name is given precedence. You can still use the index in a query expression by using the special identifier `‘index’`. You can also use the `levels` of a `DataFrame` with a `MultiIndex` as if they were columns in the frame. If the `levels` of the `MultiIndex` are unnamed, you can refer to them using special names. The convention is `ilevel_0`, which means “index level 0” for the 0th level of the index. `query()` also supports special use of Python’s `in` and `not in` comparison operators, providing a succinct syntax for calling the `isin` method of a Series or `DataFrame`. Comparing a list of values to a column using `==/!=` similarly to `in/not in`.

##### Duplicate data

If you want to identify and remove duplicate rows in a `DataFrame`, there are two methods that will help: `duplicated` and `drop_duplicates`. Each takes as an argument the columns to use to identify duplicated rows `subsets`. `duplicated` returns a boolean vector whose length is the number of rows, and which indicates whether a row is duplicated.`drop_duplicates` removes duplicate rows. By default, the first observed row of a duplicate set is considered unique, but each method has a `keep` parameter to specify targets to be kept. keep='first' (default): mark / drop duplicates except for the first occurrence; keep='last': mark / drop duplicates except for the last occurrence; keep=False: mark / drop all duplicates. To drop duplicates by index value, use `Index.duplicated` then perform slicing. The same set of options are available for the `keep` parameter.

```python
df3.index.duplicated();df3[~df3.index.duplicated()]
```

Each of Series or `DataFrame` have a `get` method which can return a default value.

###### Index objects

Indexes are “mostly immutable”, but it is possible to set and change their metadata, like the index name or, for `MultiIndex`, levels and codes. You can use the `rename, set_names, set_levels`, and `set_codes` to set these attributes directly. They default to returning a copy; however, you can specify `inplace=True` to have the data change in place. The two main operations are union (|) and intersection (&). These can be directly called as instance methods or used via overloaded operators. Difference is provided via the `.difference` method. When performing `Index.union()` between indexes with different `dtypes`, the indexes must be cast to a common `dtype`. `DataFrame` has a `set_index()` method which takes a column name or a list of column names. To create a new, re-indexed `DataFrame`. As a convenience, there is a new function on `DataFrame` called `reset_index()` which transfers the index values into the `DataFrame’s` columns and sets a simple integer index. This is the inverse operation of `set_index()`. You can use the `level` keyword to remove only a portion of the index.

```python
df.rename(columns={'A':'a','B':'b'});df.rename(str.lower, axis='columns')
dfmi.loc[:, ('one', 'second')] = value
dfmi.loc.__setitem__((slice(None), ('one', 'second')), value)
dfmi['one']['second'] = value
dfmi.__getitem__('one').__setitem__('second', value)
```

it’s very hard to predict whether it will return a view or a copy (it depends on the memory layout of the array, about which pandas makes no guarantees), and therefore whether the `__setitem__` will modify `dfmi` or a temporary object that gets thrown out immediately afterward. 

#### $\text{MultiIndex}$

For more flexibility in how the index is constructed, you can instead use the class method constructors available in the ``pd.MultiIndex``.

pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])

You can construct it from a list of tuples giving the multiple index values of each point:

pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])

You can even construct it from a Cartesian product of single indices:

pd.MultiIndex.from_product([['a', 'b'], [1, 2]])

Similarly, you can construct the ``MultiIndex`` directly using its internal encoding by passing ``levels`` (a list of lists containing available index values for each level) and ``labels`` (a list of lists that reference these labels):

pd.MultiIndex(levels=[['a', 'b'], [1, 2]],
              labels=[[0, 0, 1, 1], [0, 1, 0, 1]])

Any of these objects can be passed as the ``index`` argument when creating a ``Series`` or ``Dataframe``, or be passed to the ``reindex`` method of an existing ``Series`` or ``DataFrame``.



Sometimes it is convenient to name the levels of the ``MultiIndex``.
This can be accomplished by passing the ``names`` argument to any of the above ``MultiIndex`` constructors, or by setting the ``names`` attribute of the index after the fact:

Remember that columns are primary in a ``DataFrame``, and the syntax used for multiply indexed ``Series`` applies to the columns.

Working with slices within these index tuples is not especially convenient; trying to create a slice within a tuple will lead to a syntax error:
`health_data.loc[(:, 1), (:, 'HR')]`

You could get around this by building the desired slice explicitly using Python's built-in ``slice()`` function, but a better way in this context is to use an ``IndexSlice`` object, which Pandas provides for precisely this situation.



Earlier, we briefly mentioned a caveat, but we should emphasize it more here.
*Many of the ``MultiIndex`` slicing operations will fail if the index is not sorted.*
Let's take a look at this here.

We'll start by creating some simple multiply indexed data where the indices are *not lexographically sorted*:

Although it is not entirely clear from the error message, this is the result of the MultiIndex not being sorted.
For various reasons, partial slices and other similar operations require the levels in the ``MultiIndex`` to be in sorted (i.e., lexographical) order.
Pandas provides a number of convenience routines to perform this type of sorting; examples are the ``sort_index()`` and ``sortlevel()`` methods of the ``DataFrame``.
We'll use the simplest, ``sort_index()``, here:

All of the `MultiIndex` constructors accept a `names` argument which stores string names for the levels themselves. If no names are provided, `None` will be assigned. The method `get_level_values()` will return a vector of the labels for each location at a particular level. The `MultiIndex` keeps all the defined `levels` of an index, even if they are not actually used. To reconstruct the `MultiIndex` with only the used levels, the `remove_unused_levels()` method may be used. Operations between differently-indexed objects having `MultiIndex` on the axes will work as you expect; data alignment will work the same as an Index of tuples. The `reindex()` method of `Series/DataFrames` can be called with another `MultiIndex`, or even a list or array of tuples. In general, `MultiIndex` keys take the form of tuples.

```python
df.loc[('bar', 'two')];df.loc[('bar', 'two'), 'A']
df.loc[('baz', 'two'):('qux', 'one')]
#slice a MultiIndex by providing multiple indexers.
dfmi.loc[(slice('A1', 'A3'), slice(None), ['C1', 'C3']), :]
idx = pd.IndexSlice
dfmi.loc[idx[:, :, ['C1', 'C3']], idx[:, 'foo']]
```

###### Cross-section

The `xs()` method of `DataFrame` additionally takes a level argument to make selecting data at a particular level of a `MultiIndex` easier.

```python
df.xs('one', level='second')
df.xs(('one', 'bar'), level=('second', 'first'), axis=1)
```

Using the parameter `level` in the `reindex()` and `align()` methods of pandas objects is useful to broadcast values across a level. The `swaplevel()` method can switch the order of two levels. The `reorder_levels()` method generalizes the `swaplevel` method, allowing you to permute the hierarchical index levels in one step. The `rename_axis()` method is used to rename the name of a Index or `MultiIndex`. For `MultiIndex-ed` objects to be indexed and sliced effectively, they need to be sorted. As with any index, you can use `sort_index()`. You may also pass a `level` name to `sort_index` if the `MultiIndex` levels are named. Indexing will work even if the data are not sorted, but will be rather inefficient. It will also return a copy of the data rather than a view. The `is_lexsorted()` method on a `MultiIndex` shows if the index is sorted, and the `lexsort_depth` property returns the sort depth.

###### Take methods

pandas Index, Series, and `DataFrame` also provides the `take()` method that retrieves elements along a given axis at the given indices. The given indices must be either a list or an ndarray of integer index positions. take will also accept negative integers as relative positions to the end of the object.

```python
positions = [0, 9, 3]
ser.take(positions); index[positions]
df.take([0, 2], axis=1)
```

###### Index types

`CategoricalIndex` is a type of index that is useful for supporting indexing with duplicates. This is a container around a Categorical and allows efficient indexing and storage of an index with a large number of duplicated elements. The `CategoricalIndex` is preserved after indexing. Sorting the index will sort by the order of the categories. `Groupby` operations on the index will preserve the index nature as well. `Reindexing` operations will return a resulting index based on the type of the passed indexer. Passing a list will return a plain-old Index; indexing with a Categorical will return a `CategoricalIndex`, indexed according to the categories of the passed Categorical `dtype`. 
`Int64Index` is a fundamental basic index in pandas. This is an immutable array implementing an ordered, sliceable set. `RangeIndex` is a sub-class of `Int64Index` providing the default index for all `NDFrame` objects. `RangeIndex` is an optimized version of `Int64Index` that can represent a monotonic ordered set.
By default a `Float64Index` will be automatically created when passing floating, or mixed-integer-floating values in index creation. This enables a pure label-based slicing paradigm that makes `[],ix,loc` for scalar indexing and slicing work exactly the same.

If we need intervals on a regular frequency, we can use the `interval_range()` function to create an `IntervalIndex` using various combinations of `start`, `end`, and `periods`. The default frequency for `interval_range` is a 1 for numeric intervals, and calendar day for `datetime`-like intervals. The `freq` parameter can used to specify non-default frequencies.

 In pandas, our general viewpoint is that labels matter more than integer locations. Therefore, with an integer axis index *only* label-based indexing is possible with the standard tools like `.loc`. If the index of a Series or `DataFrame` is monotonically increasing or decreasing, then the bounds of a label-based slice can be outside the range of the index. Monotonicity of an index can be tested with the `is_monotonic_increasing()` and `is_monotonic_decreasing()` attributes.

### 下标存取

#### []操作符

对于`Index`对象，可以通过`[]`来选取数据，它**类似于一维`ndarray`的索引**。下标可以为下列几种下标对象：一个整数下标。此时返回对应的`label`；一个整数`slice`。此时返回对应的`Index`；一个`array-like`对象（元素可以为下标或者布尔值）。此时返回对应的`Index`；由`None`组成的二元组，其中`None`相当于新建一个轴。它并没有将`Index` 转换成`MultiIndex`，只是将`Index`内部的数据数组扩充了一个轴。

```python
idx = pd.Index(list('abcd'), name = 'index')
index[None], idx[0, None], idx[:, None],idx[None, :]
idex[[0, 1, 2, 3]], idx[np.array([1,2,3])], 
idx[np.array([[0, 1], [1, 2]])]
```

对于`Series`对象，可以通过`[]`来选取数据，它类似于一维`ndarray`的索引。下标可以为下列几种下标对象：一个整数下标/一个属性（属性名为某个`label`）/字典索引（键为`label`）：返回对应的数值；一个整数切片/一个`label`切片：返回对应的`Series`。注意：`label`切片同时包含了起始`label`和终止`label`；一个整数`array-like`/一个`label array-like`/一个布尔`ndarray`：返回对应的`Series`；一个二维整数`array-like`/二维`label array-like`：返回对应值组成的二维`ndarray`

对于`DataFrame`对象，可以通过`[]`来选取数据。下标可以为下列几种下标对象：一个属性属性名为某个`column label`/字典索引键为`column label`：返回对应的列对应的`Series`；一个整数切片/一个`row label`切片：返回对应的行组成的`DataFrame`。注意：`label`切片同时包含了起始`label`和终止`label`；一个一维`label array-like`:返回对应的列组成的`DataFrame`；一个布尔数组：返回数组中`True`对应的行组成的`DataFrame`；一个布尔`DataFrame`：将该布尔`DataFrame`中的`False`对应的元素设置为`NaN`（布尔`DataFrame`中没有出现的值为`False`）

对于`Series/DataFrame`切片方式的索引，返回的结果与原始对象共享基础数据。对于采用其他方式的索引，返回的结果并不与元素对象共享基础数据。

对于`DataFrame`的赋值与列删除：将列表或者数组赋值给某个列时，其长度必须跟`DataFrame`的行数匹配；将标量赋值给某个列时，会将标量扩充；将`Series`赋值给某个列时，会精确匹配`DataFrame`的索引。如果`DataFrame`中某个`label`在`Series`中找不到，则赋值`NaN`；为不存在的列赋值会创建出一个新列（必须用字典的形式，不能用属性赋值的形式）；关键字`del`用于删除列（必须用字典的形式，不能用属性赋值的形式）

对于`Series`的赋值与删除：对于单个索引或者切片索引，要求右侧数值的长度与左侧相等；为不存在的`label`赋值会创建出一个新行（**必须用字典的形式，不能用属性赋值的形式**）；关键字`del`用于删除行（必须用字典的形式，不能用属性赋值的形式）

如果`Series/DataFrame`的索引有重复`label`，则数据的选取行为将有所不同：如果索引对应多个`label`，则`Series`返回一个`Sereis`，`DataFrame`返回一个`DataFrame`；如果索引对应单个`label`，则`Series`返回一个标量值，`DataFrame`返回一个`Series`。你可以通过`Index.is_unique`属性得知索引是否有重复的。

对于`[]`、字典索引、属性索引或者`.loc/.ix`存取器，结论如上所述；对于`.at`存取器：如果索引对应单个`label`，索引结果正常。如果索引对应多个`label`，则`Series`返回一个一维`ndarray`；`DataFrame`则抛出异常。

#### loc/iloc存取器

对于`Series`， `.loc[]`的下标对象可以为：单个`label`，此时返回对应的值；`label`的`array-like`、`label slice`以及布尔`array-like`：返回对应值组成的`Series`

对于`DataFrame`，`.loc[]`的下标对象是一个元组，其中两个元素分别与`DataFrame`的两个轴对应。如果下标不是元组，则该下标对应的是第0轴，第一轴为默认值`:`；每个轴的下标都支持单个`label`、`label array-like`、`label slice`、布尔`array-like`；若获取的是某一列或者某一行，则返回的是`Series`；若返回的是多行或者多列，则返回的是`DataFrame`；如果返回的是某个值，则是普通的标量。

`.iloc[]`和`.loc[]`类似，但是`.iloc[]`使用整数下标，而不是使用`label`。

`Index`对象不能使用`loc/iloc/ix`存取器；对于`.loc/.iloc`：如果某轴的索引为`array-like`或者布尔`array-like`，则返回的结果与原来的对象不再共享基础数据。如果轴的索引全部都是`slice`或者单个整数、单个`label`，则返回的结果与原来的对象共享基础数据。

对于`DataFrame`，`.lookup(row_labels, col_labels)`类似于：`.loc[row_labels, col_labels]`，但是`.lookup`返回的是一维`ndarray`。要求`row_labels`和`col_labels`长度相同。`(row_labels[0],col_labels[0]`决定了结果中第一个元素的位置，...`(row_labels[i],col_labels[i]`决定了结果中第 `i+1`个元素的位置。

`DataFrame.get_value(index, col, takeable=False)`等价于`.loc[index, col]`，它返回单个值。而`Series.get_value(label, takeable=False)`等价于`.loc[label]`，它也返回单个值。`.get(key[, default])`方法与字典的`get()`方法的用法相同。对于`DataFrame`，`key`为`col_label` 

#### query 方法

对于`DataFrame`，当需要根据一定的条件对行进行过滤时，通常可以先创建一个布尔数组，然后使用该数组获取`True`对应的行。另一个方案就是采用`query(expr, inplace=False, **kwargs)`方法：`expr`是个运算表达式字符串，如`'label1 >3 and label2<5'`；表达式中的变量名表示对应的列，可以使用`not/and/or`等关键字进行向量布尔运算；如果希望在表达式中使用`Python`变量，则在变量名之前使用`@`；`inplace`是个布尔值，如果为`True`，则原地修改。否则返回一份拷贝。

#### 多级索引

对于`.loc/.ix/[]`，其下标可以指定多级索引中，每级索引上的标签。多级索引轴对应的下标是一个下标元组，该元组中每个元素与索引中每级索引对应；如果下标不是元组，则将其转换成长度为 1 的元组；如果元组的长度比索引的层数少，则在其后面补充`slice(None)`

```python
idx = pd.MultiIndex(levels=[['a', 'b'], ['c', 'd']], 
                   labels=[[0, 0, 1, 1],[0, 1, 0, 1]], names=['1', '2'])
s = pd.Series([0, 3, 5, 9], index = idx, name = 'series')
s.loc[('a', 'c')]
df = pd.DataFrame({'c1':[1, 2, 3,4], 'c2':[5, 6, 7, 8]}, index = idx)
df.loc[('a', slice(None)), ['c1', 'c2']]
```

### 