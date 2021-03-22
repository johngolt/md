There are five preset `seaborn` themes: `darkgrid, whitegrid, dark, white, and ticks`. The default theme is `darkgrid`. Both the `white` and `ticks` styles can benefit from removing the top and right axes spines

```pyhon
sns.set_style("whitegrid")
```

控制绘图元素的尺度，使得相同的代码对于不同的场景展示的图像大小适当。The four preset contexts, are `paper, notebook, talk`, and `poster`. The notebook style is the default.

```python
sns.set_context("paper")
```

#### 数值特征的分布

##### 数值$\times$ 数值 散点图

```python
tips = sns.load_dataset("tips")
#hue, style可以分类展示不同类别情况，size, col, row
# kind='line' to draw a line plot. scatter
sns.relplot(x="total_bill", y="tip", data=tips);
```

##### 数值$\times$ 数值 分布图

The `jointplot()` function uses a `JointGrid` to manage the figure.  `jointplot()` returns the `JointGrid` object after plotting

```python
mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])
# kind='hex', 'kde'
sns.jointplot(x="x", y="y", data=df);
```

##### 单数值特征可视化

The most convenient way to take a quick look at a univariate distribution in `seaborn` is the `distplot` function, this will draw a histogram and fit a kernel density estimate. 

```python
x = np.random.normal(size=100)
#bins分箱的数目。
#fit kde与fit设置的分布进行对比
sns.distplot(x);
```

##### 多数值特征分布

To plot multiple pairwise bivariate distributions in a dataset, you can use the `pairplot()` function. This creates a matrix of axes and shows the relationship for each pair of columns in a `DataFrame`. the `pairplot()` function is built on top of a `PairGrid` object

```python
iris = sns.load_dataset("iris")
sns.pairplot(iris);

g = sns.PairGrid(iris)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter);

g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
```

##### 相关系数可视化

```python
uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(uniform_data)
'''fmt : string, optional,String formatting code to use when adding annotations.'''
```

#### 类别特征分布

##### 类别$\times$ 数值 散点图

```python
tips = sns.load_dataset("tips")
# kind = strip, swarm运行速度很慢
sns.catplot(x="day", y="total_bill", data=tips);
```

##### 类别$\times$ 数值 分布图

```python
#kind = box, violin, boxen
sns.catplot(x="day", y="total_bill", kind="box", data=tips);
sns.catplot(data=iris, orient="h", kind="box");
```

##### 类别$\times$ 数值统计值可视化

```python
titanic = sns.load_dataset("titanic")
# kind='bar', point
# estimator数值特征的估计函数
sns.catplot(x="sex", y="survived",kind="bar", data=titanic);
```

##### 单类别特征可视化

```python
sns.catplot(x="deck", kind="count", data=titanic);
```

#### multi-plot grids

`FacetGrid` can be drawn with up to three dimensions: `row, col, and hue`. The first two have obvious correspondence with the resulting array of axes; think of the hue variable as a third dimension along a depth axis, where different levels are plotted with different colors.

The main approach for visualizing data on this grid is with the `FacetGrid.map()` method. Provide it with a plotting function and the name(s) of variable(s) in the `dataframe` to plot. 

```python
g = sns.FacetGrid(tips, col="time")
g.map(plt.hist, "tip");

g = sns.FacetGrid(tips, col="sex", hue="smoker")
g.map(plt.scatter, "total_bill", "tip", alpha=.7)
g.add_legend();
```

The most general is `FacetGrid.set()`, and there are other more specialized methods like `FacetGrid.set_axis_labels()`, which respects the fact that interior facets do not have axis labels. 

```python
g.set_axis_labels("Total bill (US Dollars)", "Tip");
g.set(xticks=[10, 30, 50], yticks=[2, 6, 10]);
g.fig.subplots_adjust(wspace=.02, hspace=.02);
```

##### Conditional small multiples

It must plot onto the “currently active” `matplotlib` Axes. This will be true of functions in the `matplotlib.pyplot`  name space, and you can call `plt.gca` to get a reference to the current Axes if you want to work directly with its methods.
It must accept the data that it plots in positional arguments. Internally, `FacetGrid` will pass a Series of data for each of the named positional arguments passed to `FacetGrid.map()`.
It must be able to accept color and label keyword arguments, and, ideally, it will do something useful with them. In most cases, it’s easiest to catch a generic dictionary of `**kwargs` and pass it along to the underlying plotting function.

```python
from scipy import stats
def quantile_plot(x, **kwargs):
    qntls, xr = stats.probplot(x, fit=False)
    plt.scatter(xr, qntls, **kwargs)

g = sns.FacetGrid(tips, col="sex", height=4)
g.map(quantile_plot, "total_bill");
```

