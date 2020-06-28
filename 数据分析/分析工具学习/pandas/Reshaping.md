#### Reshaping and pivot tables

![](D:/MarkDown/picture/1/154.png)

`pivot()` will error with a `ValueError`: Index contains duplicate entries, cannot reshape if the index/column pair is not unique. In this case, consider using `pivot_table()` which is a generalization of pivot that can handle duplicate values for one index/column pair.

If the columns have a `MultiIndex`, you can choose which level to stack. The stacked level becomes the new lowest level in a `MultiIndex` on the columns. With a “stacked” `DataFrame` or `Series` (having a `MultiIndex` as the `index`), the inverse operation of `stack` is `unstack`, which by default unstacks the **last level**:

![](D:/MarkDown/picture/1/155.png)

![](D:/MarkDown/picture/1/156.png)

If the indexes have names, you can use the level names instead of specifying the level numbers. Notice that the `stack` and `unstack` methods implicitly sort the index levels involved. Hence a call to `stack` and then `unstack`, or vice versa, will result in a **sorted** copy of the original `DataFrame` or `Series`. You may also stack or unstack more than one level at a time by passing a list of levels, in which case the end result is as if each level in the list were processed individually. The list of levels can contain either level names or level numbers. Unstacking can result in missing values if subgroups do not have the same set of labels. By default, missing values will be replaced with the default fill value for that data type, `NaN` for float.

![](D:/MarkDown/picture/1/157.png)

![](D:/MarkDown/picture/1/158.png)

The top-level `melt()` function and the corresponding `DataFrame.melt()` are useful to massage a DataFrame into a format where one or more columns are identifier variables, while all other columns, considered measured variables, are “unpivoted” to the row axis, leaving just two non-identifier columns, “variable” and “value”. The names of those columns can be customized by supplying the `var_name` and `value_name` parameters.

###### Pivot tables

The difference between pivot tables and ``GroupBy`` can sometimes cause confusion; it helps me to think of pivot tables as essentially a *multidimensional* version of ``GroupBy`` aggregation.
That is, you split-apply-combine, but both the split and the combine happen across not a one-dimensional index, but across a two-dimensional grid.

The function `pivot_table()` can be used to create spreadsheet-style pivot tables. It takes a number of arguments: data: a DataFrame object. values: a column or a list of columns to aggregate. 对values进行聚合时，是对每一个value进行单独的聚合，等价于分组之后，单独取每个特征进行聚合，然后在组合起来。并不是对采用了index和columns分组之后，使用values的DataFrame来进行聚合计算，`aggfunc`的输入为`Series`。`index`: a column, Grouper, array which has the same length as data, or list of them. `columns`: a column, Grouper, array which has the same length as data, or list of them.  `aggfunc`: function to use for aggregation. At times it's useful to compute totals along each grouping. This can be done via the ``margins`` keyword. The margin label can be specified with the ``margins_name`` keyword, which defaults to ``"All"``.

**1. 透视表**

​         1.1. pivot

​         1.2. pivot_table

​         1.3. crosstab(交叉表）

- -  **2. 其他变形方法**

  -  2.1. melt函数

  -  2.2. 压缩与展开

​        **3. 哑变量与因子化**

- -   3.1. Dummy Variable（哑变量）

  -   3.2. factorize方法