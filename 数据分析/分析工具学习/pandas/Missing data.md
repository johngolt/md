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

##### 