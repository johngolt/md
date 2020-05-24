The dots (...) represent as many colons as needed to produce a complete indexing tuple. if x is an array with 5 axes, then: x[1,2,...] is equivalent to x[1,2,:,:,:], and x[4,...,5,:] to x[4,:,:,5,:].

However, if one wants to perform an operation on each element in the array, one can use the `flat` attribute which is an iterator over all the elements of the array. The order of the elements in the array resulting from `ravel()` is normally “C-style”, that is, the rightmost index “changes the fastest”, so the element after a[0,0] is a[0,1]. If the array is reshaped to some other shape, again the array is treated as “C-style”. NumPy normally creates arrays stored in this order, so `ravel()` will usually not need to copy its argument, but if the array was made by taking slices of another array or created with unusual options, it may need to be copied. The functions `ravel()` and `reshape()` can also be instructed, using an optional argument, to use FORTRAN-style arrays, in which the leftmost index changes the fastest.

The `reshape` function returns its argument with a modified shape, whereas the `ndarray.resize` method modifies the array itself. The function `column_stack` stacks `1D` arrays as columns into a `2D` array，arrays的维度可以不保持一致. It is equivalent to `hstack` only for `2D` arrays. On the other hand, the function `row_stack` is equivalent to `vstack` for any input arrays. In general, for arrays of with more than two dimensions, `hstack` stacks along their second axes, `vstack` stacks along their first axes, and `concatenate` allows for an optional arguments giving the number of the axis along which the concatenation should happen.
Using `hsplit`, you can split an array along its horizontal axis, either by specifying the number of equally shaped arrays to return, or by specifying the columns after which the division should occur:
`vsplit` splits along the vertical axis, and `array_split` allows one to specify along which axis to split.

Different array objects can share the same data. The view method creates a new array object that looks at the same data. Slicing an array returns a view of it. The `copy` method makes a complete copy of the array and its data.
Sometimes `copy` should be called after slicing if the original array is not required anymore. 
Broadcasting allows universal functions to deal in a meaningful way with inputs that do not have exactly the same shape.

##### Data types

There are 5 basic numerical types representing booleans (bool), integers (int), unsigned integers (uint) floating point (float) and complex. Those with numbers in their name indicate the bitsize of the type. Data-types can be used as functions to convert python numbers to array scalars, python sequences of numbers to arrays of that type, or as arguments to the `dtype` keyword that many numpy functions or methods accept. To convert the type of an array, use the `.astype()` method  or the type itself as a function. NumPy generally returns elements of arrays as array scalars (a scalar with an associated `dtype`). Array scalars differ from Python scalars, but for the most part they can be used interchangeably. when code requires very specific attributes of a scalar or when it checks specifically whether a value is a Python scalar. Generally, problems are easily fixed by explicitly converting array scalars to Python scalars, using the corresponding Python type function. The primary advantage of using array scalars is that they preserve the array type. Therefore, the use of array scalars ensures identical behavior between arrays and scalars, irrespective of whether the value is inside an array or not. NumPy scalars also have many of the same methods arrays do.

The fixed size of NumPy numeric types may cause overflow errors when a value requires more memory than available in the data type. The behavior of NumPy and Python integer types differs significantly for integer overflows and may confuse users expecting NumPy integers to behave similar to Python’s int. Unlike NumPy, the size of Python’s int is flexible. This means Python integers may expand to accommodate any integer and will not overflow.

NumPy provides `numpy.iinfo` and `numpy.finfo` to verify the minimum or maximum values of NumPy integer and floating point values respectively

##### Array Creation

There are 5 general mechanisms for creating arrays: Conversion from other Python structures; Intrinsic `numpy` array creation objects; Reading arrays from disk, either from standard or custom formats; Creating arrays from raw bytes through the use of strings or buffers; Use of special library functions.

##### Indexing

###### Single element indexing

Single element indexing for a 1-D array is what one expects. It work exactly like that for other standard Python sequences. It is 0-based, and accepts negative indices for indexing from the end of the array. Unlike lists and tuples, numpy arrays support multidimensional indexing for multidimensional arrays. That means that it is not necessary to separate each dimension’s index into its own set of square brackets. Note that if one indexes a multidimensional array with fewer indices than dimensions, one gets a subdimensional array.  NumPy uses C-order indexing. That means that the last index usually represents the most rapidly changing memory location, unlike Fortran or IDL, where the first index represents the most rapidly changing location in memory.

###### index arrays

NumPy arrays may be indexed with other arrays or any other sequence- like object that can be converted to an array, such as lists, with the exception of tuples. For all cases of index arrays, what is returned is a copy of the original data, not a view as one gets for slices. Index arrays must be of integer type. Each value in the array indicates which value in the array to use in place of the index. Negative values are permitted and work as they do with single indices or slices. Generally speaking, what is returned when index arrays are used is an array with the same shape as the index array, but with the type and values of the array being indexed. 

###### indexing Multi-dimensional arrays

```python
y[np.array([0,2,4]), np.array([0,1,2])]
```

In this case, if the index arrays have a matching shape, and there is an index array for each dimension of the array being indexed, the resultant array has the same shape as the index arrays, and the values correspond to the index set for each position in the index arrays. In this example, the first index value is 0 for both index arrays, and thus the first value of the resultant array is y[0,0]. The next value is y[2,1], and the last is y[4,2].

If the index arrays do not have the same shape, there is an attempt to broadcast them to the same shape. If they cannot be broadcast to the same shape, an exception is raised. The broadcasting mechanism permits index arrays to be combined with scalars for other indices. The effect is that the scalar value is used for all the corresponding values of the index arrays. it is possible to only partially index an array with index arrays.

```python
y[np.array([0,2,4])]
# What results is the construction of a new array where each value of the index array selects one row from the array being indexed and the resultant array has the resulting shape
```

  Boolean arrays must be of the same shape as the initial dimensions of the array being indexed. In the most straightforward case, the boolean array has the same shape. in the boolean case, the result is a 1-D array containing all the elements in the indexed array corresponding to all the true elements in the boolean array. As with index arrays, what is returned is a copy of the data, not a view as one gets with slices. In general, when the boolean array has fewer dimensions than the array being indexed, this is equivalent to `y[b, …]`, which means `y` is indexed by `b` followed by as many as are needed to fill out the rank of `y`. Thus the shape of the result is one dimension containing the number of `True` elements of the boolean array, followed by the remaining dimensions of the array being indexed.

```python
x = np.arange(30).reshape(2,3,5)
b = np.array([[True, True, False], [False, True, True]])
x[b] # shape=(2,2,5) 
```

In effect, the slice is converted to an index array that is broadcast with the index array to produce a resultant array of shape. Likewise, slicing can be combined with broadcasted boolean indices. To facilitate easy matching of array shapes with expressions and in assignments, the `np.newaxis` object can be used within array indices to add new dimensions with a size of 1.

##### Broadcasting

When operating on two arrays, NumPy compares their shapes element-wise. It starts with the trailing dimensions, and works its way forward. Two dimensions are compatible when they are equal, or one of them is 1. If these conditions are not met, a `ValueError` exception is thrown, indicating that the arrays have incompatible shapes. The size of the resulting array is the maximum size along each dimension of the input arrays.

```python
A      (4d array):  8 x 1 x 6 x 1
B      (3d array):      7 x 1 x 5
Result (4d array):  8 x 7 x 6 x 5
```

