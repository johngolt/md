#### Indexing and selecting data

Any of the axes accessors may be the null slice `:`. Axes left out of the specification are assumed to be `:`, e.g. `p.loc['a']` is equivalent to `p.loc['a', :, :]`.

| Object Type | Selection        | Return Value Type                |
| ----------- | ---------------- | -------------------------------- |
| `Series`    | `series[label]`  | scalar value                     |
| `DataFrame` | `frame[colname]` | `Series` corresponding to column |

###### Attribute access

```python
sa = pd.Series([1, 2, 3], index=list('abc'))
sa.b
x = pd.DataFrame({'x': [1, 2, 3], 'y': [3, 4, 5]})
x.iloc[1] = {'x': 9, 'y': 99}
```

###### Slicing ranges

With Series, the syntax works exactly as with an `ndarray`, returning a slice of the values and the corresponding labels. 

```python
s[::2]
```

With `DataFrame`, slicing inside of `[]` slices the rows. This is provided largely as a convenience since it is such a common operation. 选择行

###### Selection by label

Every label asked for must be in the index, or a `KeyError` will be raised. When slicing, both the start bound AND the stop bound are included, if present in the index. Integers are valid labels, but they refer to the label and not the position.

`.loc` allows inputs are: 

- A single label
- A list or array of labels
- A slice object with labels 'a':'f', Note that contrary to usual python slices, both the start and the stop are included, when present in the index
- A boolean array
- A callable function with one argument and that returns valid output for indexing，`iloc`中使用的函数，传入参数就是前面的`df`。

```python
df1 = pd.DataFrame(np.random.randn(6, 4), index=list('abcdef'), columns=list('ABCD'))
df1.loc['A'] # A single labels
df1.loc[:,['A','B']]# a list or array of labels
df1.loc['d':, 'A':'C']## A slice object with labels
df1.loc[:, df1.loc['a'] > 0]# boolean Array.
df1.loc[lambda df: df['A'] > 0, :] # callable
```

`.loc` is strict when you present slicers that are not compatible (or convertible) with the index type. These will raise a `TypeError`. When using `.loc` with slices, 如果起始点都在index，取数返回结果则包括起始点在内的中间点会被返回，如果起始点中存在一个缺失，但是index是有序的，那么取数过程中会比较与起始点进行比较，可以正常返回；如果index是无序的则会报错。

```python
s = pd.Series(list('abcde'), index=[0, 3, 2, 5, 4])
s.sort_index().loc[1:6]
```

###### Selection by position

These are `0-based` indexing. When slicing, the start bound is *included*, while the upper bound is *excluded*. Trying to use a non-integer, even a valid label will raise an `IndexError`, `.iloc` will raise `IndexError` if a requested indexer is out-of-bounds, except slice indexers which allow out-of-bounds indexing

- `.iloc` allows inputs are: 
  - An integer
  - A list or array of integers
  - A slice object with ints
  - A boolean array
  - A callable function with one argument and that returns valid output for indexing，`iloc`中使用的函数，传入参数就是前面的`df`.

```python
df1.iloc[0]# An integer
df1.iloc[[1,2,3],:] # a list or array of integer
df1.iloc[1:5,2:4]# slice object with ints
df1.iloc[:, lambda df: [0, 1]] #callable
df1.loc[:, (df1.loc['a'] > 0).to_numpy()]# boolean array
```

###### $\text{Reindexing}$

if you want to select only valid keys, the following is idiomatic and efficient; it is guaranteed to preserve the `dtype` of the selection.

```python
s.loc[s.index.intersection([1,2,3])]  # select only valid keys
#Having a duplicated index will raise for a `.reindex()`,会报错。
```

###### Selecting random samples

```python
s = pd.Series([0, 1, 2, 3, 4, 5])## When no arguments, returns 1 row.
s.sample(n=3)#specify a number of rows
s.sample(frac=0.5)#a fraction of the rows
s.sample(n=6, replace=True)# 有放回
df2 = pd.DataFrame({'col1': [9, 8, 7, 6],'weight_column': [0.5, 0.4, 0.1, 0]})
df2.sample(n=3, weights='weight_column') # 使用列作为样本权重
df2.sample(n=1,axis=1) # 设定选择样本的行或列
df2.sample(n=2, random_state=2) # 设置随机种子
```

默认`sample`是无放回抽样，通过设置`replace`参数可以使用有放回抽样。默认所有样本被选中的概率一样可以通过设置`weights`参数来设置每个样本被选中的概率，Missing values will be treated as a weight of zero, and `inf` values are not allowed.

###### Scalar Value getting and setting

If you only want to access a scalar value, the fastest way is to use the `at` and `iat` methods, which are implemented on all of the data structures. Similarly to `loc`, at provides label based scalar lookups, while, `iat` provides integer based lookups analogously to `iloc`.

###### Boolean indexing

Another common operation is the use of boolean vectors to filter the data. 

The operators are: `|` for `or`, `&` for `and`, and `~` for `not`. These must be grouped by using parentheses, since by default Python will evaluate an expression such as `df.A > 2 & df.B < 3` as `df.A > (2 & df.B) < 3`. 

Using a boolean vector to index a Series works exactly as in a `NumPy`. You may select rows from a `DataFrame` using a boolean vector the same length as the `DataFrame’s` index. 

```python
df2 = pd.DataFrame({'a': ['one', 'one', 'two', 'three', 'two', 'one', 'six'],
 'b': ['x', 'y', 'y', 'x', 'y', 'x', 'x'], 'c': np.random.randn(7)})
criterion = df2['a'].map(lambda x: x.startswith('t'))
df2[criterion]
```

###### Indexing with `isin`

Consider the `isin()` method of Series, which returns a boolean vector that is true wherever the Series elements exist in the passed list. This allows you to select rows where one or more columns have values you want.

 `DataFrame` also has an `isin()` method. When calling `isin`, pass a set of values as either an `array` or `dict`. If values is an `array`, `isin` returns a `DataFrame` of booleans that is the same shape as the original `DataFrame`, with True wherever the element is in the sequence of values.

```python
s_mi = pd.Series(np.arange(6),index=pd.MultiIndex.from_product([[0, 1], ['a', 'b', 'c']]))
s_mi.iloc[s_mi.index.isin([(1, 'a'), (2, 'b'), (0, 'c')])]
s_mi.iloc[s_mi.index.isin(['a', 'c', 'e'], level=1)]
```

Oftentimes you’ll want to match certain values with certain columns. Just make values a `dict` where the key is the column, and the value is a list of items you want to check for. 

```python
df = pd.DataFrame({'vals': [1, 2, 3, 4], 'ids': ['a', 'b', 'f', 'n'],                  'ids2': ['a', 'n', 'c', 'n']})
values = {'ids': ['a', 'b'], 'vals': [1, 3]}
df.isin(values)
```

Combine `DataFrame’s` `isin` with the `any()` and `all()` methods to quickly select subsets of your data that meet a given criteria. 

```python
values = {'ids': ['a', 'b'], 'ids2': ['a', 'c'], 'vals': [1, 3]}
row_mask = df.isin(values).all(1)
```

###### The `where()` Method and Masking

输出的bool值的shape跟原始的`DataFrame`相同的shape., you can use the `where` method in Series and `DataFrame`. `other`参数控制`False`情况下的取值。

In addition, `where` takes an optional `other` argument for replacement of values where the condition is `False`, in the returned copy.

默认情况下`where`返回的结果是原来的`DataFrame`的副本，可以通过`inplace`来控制。

 `Where` can also accept `axis` and `level` parameters to align the input when performing the `where`. 

`Where` can accept a callable as condition and other arguments. The function must be with one argument (the calling Series or `DataFrame`) and that returns valid output as condition and other argument.

 `mask()` is the inverse boolean operation of `where`.

```python
df2.where(df2 > 0, df2['A'], axis='index')# align the input
df3.where(lambda x: x > 4, lambda x: x + 10) # callable
```

###### The `query()` Method

如果index的name和column的name重复，则column优先。可以使用`"index"`作为index在表达式中。

```python
df = pd.DataFrame(np.random.rand(n, 3), columns=list('abc'))
df.query('(a < b) & (b < c)')
# If instead you don’t want to or cannot name your index, you can use the name index in your query expression.
df.query('index < b < c')
df.query('[1, 2] in c'); df.query('ilevel_0=="red"')
df.query('b == ["a", "b", "c"]')
df.query('~bools')#negate boolean expressions with not or the ~.
```

You can also use the `levels` of a `DataFrame` with a `MultiIndex` as if they were columns in the frame. If the `levels` of the `MultiIndex` are unnamed, you can refer to them using special names. The convention is `ilevel_0`, which means “index level 0” for the 0th level of the index.

 `query()` also supports special use of Python’s `in` and `not in` comparison operators, providing a succinct syntax for calling the `isin` method of a Series or `DataFrame`. 

Comparing a list of values to a column using `==/!=` similarly to `in/not in`.

```python
df.query('b == ["a", "b", "c"]')
```

##### Duplicate data

If you want to identify and remove duplicate rows in a `DataFrame`, there are two methods that will help: `duplicated` and `drop_duplicates`. 

Each takes as an argument the columns to use to identify duplicated rows `subsets`. `duplicated` returns a boolean vector whose length is the number of rows, and which indicates whether a row is duplicated.

`drop_duplicates` removes duplicate rows. By default, the first observed row of a duplicate set is considered unique, but each method has a `keep` parameter to specify targets to be kept. keep='first' (default): mark / drop duplicates except for the first occurrence; keep='last': mark / drop duplicates except for the last occurrence; keep=False: mark / drop all duplicates. To drop duplicates by index value, use `Index.duplicated` then perform slicing. The same set of options are available for the `keep` parameter.

```python
df3.index.duplicated();df3[~df3.index.duplicated()]
```

Each of Series or `DataFrame` have a `get` method which can return a default value.类似于字典

###### Index objects

index是不可变类型，但是可以修改index的元数据如index name或者`MultiIndex`的levels和codes。You can use the `rename, set_names, set_levels`, and `set_codes` to set these attributes directly. 默认返回副本，可以通过`inplace`控制。

The two main operations are union (|) and intersection (&). These can be directly called as instance methods or used via overloaded operators. Difference is provided via the `.difference` method. When performing `Index.union()` between indexes with different `dtypes`, the indexes must be cast to a common `dtype`.

 `DataFrame` has a `set_index()` method which takes a column name or a list of column names.

To create a new, re-indexed `DataFrame`. As a convenience, there is a new function on `DataFrame` called `reset_index()` which transfers the index values into the `DataFrame’s` columns and sets a simple integer index. This is the inverse operation of `set_index()`. You can use the `level` keyword to remove only a portion of the index.

```python
df.rename(columns={'A':'a','B':'b'});df.rename(str.lower, axis='columns')
dfmi.loc[:, ('one', 'second')] = value
dfmi.loc.__setitem__((slice(None), ('one', 'second')), value)
dfmi['one']['second'] = value
dfmi.__getitem__('one').__setitem__('second', value)
```

#### $\text{MultiIndex}$

```python
# MultiIndex的构造方法。
pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])
pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
pd.MultiIndex.from_product([['a', 'b'], [1, 2]])
pd.MultiIndex(levels=[['a', 'b'], [1, 2]],
              labels=[[0, 0, 1, 1], [0, 1, 0, 1]])
```

Sometimes it is convenient to name the levels of the ``MultiIndex``.
This can be accomplished by passing the ``names`` argument to any of the above ``MultiIndex`` constructors, or by setting the ``names`` attribute of the index after the fact

###### Using Slicers

You could get around this by building the desired slice explicitly using Python's built-in ``slice()`` function, but a better way in this context is to use an ``IndexSlice`` object, which Pandas provides for precisely this situation.

```python
dfmi.loc[(slice(None),slice(None),['C1','C3']), (slice(None), 'foo')]
idx = pd.IndexSlice
dfmi.loc[idx[:, :, ['C1', 'C3']], idx[:, 'foo']]
```




Many of the ``MultiIndex`` slicing operations will fail if the index is not sorted.


For various reasons, partial slices and other similar operations require the levels in the ``MultiIndex`` to be in sorted order.
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

The `xs()` method of `DataFrame` 通过参数`level`来控制从特定的`level`选择数据。

```python
df.xs('one', level='second')
df.T.xs(('one', 'bar'), level=('second', 'first'), axis=1) # 选择column
```

###### Advanced reindexing and alignment

Using the parameter `level` in the `reindex()` and `align()` methods of pandas objects is useful to broadcast values across a level. 

###### Reordering levels

The `swaplevel()` method can switch the order of two levels. 

The `reorder_levels()` method generalizes the `swaplevel` method, allowing you to permute the hierarchical index levels in one step. 

###### Renames names of index

The `rename()`method is used to rename the labels of a `MultiIndex`, and is typically used to rename the columns of a `DataFrame`. The `rename_axis()` method is used to rename the name of a Index or `MultiIndex`.

###### Sorting a Multi-Index

 For `MultiIndex-ed` objects to be indexed and sliced effectively, they need to be sorted. As with any index, you can use `sort_index()`. You may also pass a `level` name to `sort_index` if the `MultiIndex` levels are named. Indexing will work even if the data are not sorted, but will be rather inefficient. It will also return a copy of the data rather than a view. The `is_lexsorted()` method on a `MultiIndex` shows if the index is sorted, and the `lexsort_depth` property returns the sort depth.

###### Take methods

pandas Index, Series, and `DataFrame` also provides the `take()` method that retrieves elements along a given axis at the given indices. The given indices must be either a list or an ndarray of integer index positions. take will also accept negative integers as relative positions to the end of the object.

```python
positions = [0, 9, 3]
ser.take(positions); index[positions]
df.take([0, 2], axis=1)
```

##### Index types

###### Categorical Index

`CategoricalIndex` is a type of index that is useful for supporting indexing with duplicates. 

```python
from pandas.api.types import CategoricalDtype
df = pd.DataFrame({'A': np.arange(6), 'B': list('aabbca')})
df['B'] = df['B'].astype(CategoricalDtype(list('cab')))
```

Sorting the index will sort by the order of the categories. indexing 和`groupby`都不改变`CategoricalIndex`的类型。`Reindexing` operations will return a resulting index based on the type of the passed indexer.  

###### Int64 Index

`Int64Index` is a fundamental basic index in pandas. This is an immutable array implementing an ordered, sliceable set.

 `RangeIndex` is an optimized version of `Int64Index` that can represent a monotonic ordered set.

By default a `Float64Index` will be automatically created when passing floating, or mixed-integer-floating values in index creation. This enables a pure label-based slicing paradigm that makes `[],ix,loc` for scalar indexing and slicing work exactly the same.

###### Interval Index

If we need intervals on a regular frequency, we can use the `interval_range()` function to create an `IntervalIndex` using various combinations of `start`, `end`, and `periods`. The default frequency for `interval_range` is a 1 for numeric intervals, and calendar day for `datetime`-like intervals. The `freq` parameter can used to specify non-default frequencies.



 In pandas, our general viewpoint is that labels matter more than integer locations. Therefore, with an integer axis index *only* label-based indexing is possible with the standard tools like `.loc`. If the index of a Series or `DataFrame` is monotonically increasing or decreasing, then the bounds of a label-based slice can be outside the range of the index. Monotonicity of an index can be tested with the `is_monotonic_increasing()` and `is_monotonic_decreasing()` attributes.
