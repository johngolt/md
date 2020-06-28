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