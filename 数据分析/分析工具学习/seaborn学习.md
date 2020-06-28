#### Visualizing statistical relationships

Statistical analysis is a process of understanding how variables in a dataset relate to each other and how those relationships depend on other variables.

We will discuss three `seaborn` functions in this tutorial. The one we will use most is `relplot()`. This is a figure-level function for visualizing statistical relationships using two common approaches: scatter plots and line plots. `relplot()` combines a `FacetGrid` with one of two axes-level functions: `scatterplot() `, `lineplot()`. As we will see, these functions can be quite illuminating because they use simple and easily-understood representations of data that can nevertheless represent complex dataset structures. They can do so because they plot two-dimensional graphics that can be enhanced by mapping up to three additional variables using the semantics of` hue, size, and style`.

##### Relating variable with scatter plots

The scatter plot is a mainstay of statistical visualization. It depicts the joint distribution of two variables using a cloud of points, where each point represents an observation in the dataset. This depiction allows the eye to infer a substantial amount of information about whether there is any meaningful relationship between them. the hue semantic was categorical, the default qualitative palette was applied. If the hue semantic is numeric specifically, if it can be cast to float, the default coloring switches to a sequential palette

##### Emphasizing continuity with line plots

With some datasets, you may want to understand changes in one variable as a function of time, or a similarly continuous variable. In this situation, a good choice is to draw a line plot. In `seaborn`, this can be accomplished by the `lineplot()` function, either directly or with `relplot()` by setting `kind="line"`. More complex datasets will have multiple measurements for the same value of the `x` variable. The default behavior in `seaborn` is to aggregate the multiple measurements at each `x` value by plotting the mean and the 95% confidence interval around the mean.  `relplot()` is based on the `FacetGrid`, this is easy to do. To show the influence of an additional variable, instead of assigning it to one of the semantic roles in the plot, use it to “facet” the visualization. This means that you make multiple axes and plot subsets of the data on each of them. You can also show the influence two variables this way: one by faceting on the columns and one by faceting on the rows. Remember that the size `FacetGrid` is parameterized by the height and aspect ratio of each facet. 

```python
import seaborn as sns
sns.relplot(x="total_bill", y="tip", hue="smoker",col="time", data=tips);

sns.relplot(x="timepoint", y="signal", hue="subject",col="region", row="event", height=3,
            kind="line", estimator=None, data=fmri);

sns.relplot(x="timepoint", y="signal", hue="event", style="event",col="subject",                          col_wrap=5,height=3, aspect=.75, linewidth=2.5,
            kind="line", data=fmri.query("region == 'frontal'"));
```

#### Plotting with categorical data

In the relational plot tutorial we saw how to use different visual representations to show the relationship between multiple variables in a dataset. In the examples, we focused on cases where the main relationship was between two numerical variables. If one of the main variables is “categorical” (divided into discrete groups) it may be helpful to use a more specialized approach to visualization.

In `seaborn`, there are several different ways to visualize a relationship involving categorical data. There are a number of axes-level functions for plotting categorical data in different ways and a figure-level interface, `catplot()`, that gives unified higher-level access to them. It is helpful to think of the different categorical plot kinds as belonging to three different families.

Categorical `scatterplots`: `stripplot()`、`swarmplot() with kind="swarm"`
Categorical distribution plots: `boxplot() with kind="box"`、 `violinplot() with kind="violin"`、 `boxenplot() with kind="boxen"`
Categorical estimate plots:`pointplot() with kind="point"`、`barplot() with kind="bar"`、`countplot() with kind="count"`
These families represent the data using different levels of granularity. When knowing which to use, you will have to think about the question that you want to answer. 

##### Categorical `scatterplots`

The default representation of the data in `catplot()` uses a `scatterplot`. There are actually two different categorical scatter plots in `seaborn`. They take different approaches to resolving the main challenge in representing categorical data with a scatter plot, which is that all of the points belonging to one category would fall on the same position along the axis corresponding to the categorical variable. The approach used by `stripplot()`, which is the default “kind” in `catplot()` is to adjust the positions of points on the categorical axis with a small amount of random “jitter”. The second approach adjusts the points along the categorical axis using an algorithm that prevents them from overlapping. It can give a better representation of the distribution of observations, although it only works well for relatively small datasets. This kind of plot is sometimes called a `“beeswarm”` and is drawn in `seaborn` by `swarmplot()`, which is activated by setting kind="swarm" in `catplot()`.

Unlike with numerical data, it is not always obvious how to order the levels of the categorical variable along its axis. In general, the `seaborn` categorical plotting functions try to infer the order of categories from the data. If your data have a pandas Categorical `datatype`, then the default order of the categories can be set there. If the variable passed to the categorical axis looks numerical, the levels will be sorted. But the data are still treated as categorical and drawn at ordinal positions on the categorical axes even when numbers are used to label them

```python
sns.catplot(x="size", y="total_bill", kind="swarm",
            data=tips.query("size != 3"));
```

The other option for choosing a default ordering is to take the levels of the category as they appear in the dataset. The ordering can also be controlled on a plot-specific basis using the order parameter. This can be important when drawing multiple categorical plots in the same figure.

```python
sns.catplot(x="smoker", y="tip", order=["No", "Yes"], data=tips);
```

put the categorical variable on the vertical axis particularly when the category names are relatively long or there are many categories. To do this, swap the assignment of variables to axes

```python
sns.catplot(x="total_bill", y="day", hue="time", kind="swarm", data=tips);
```

##### Distributions of observations within categories

As the size of the dataset grows, categorical scatter plots become limited in the information they can provide about the distribution of values within each category. When this happens, there are several approaches for summarizing the distributional information in ways that facilitate easy comparisons across the category levels

The first is the familiar `boxplot()`. This kind of plot shows the three quartile values of the distribution along with extreme values. The “whiskers” extend to points that lie within `1.5 IQRs` of the lower and upper quartile, and then observations that fall outside this range are displayed independently. This means that each value in the `boxplot` corresponds to an actual observation in the data.
A related function, `boxenplot()`, draws a plot that is similar to a box plot but optimized for showing more information about the shape of the distribution. It is best suited for larger datasets:
A different approach is a `violinplot()`, which combines a `boxplot` with the kernel density estimation procedure. This approach uses the kernel density estimate to provide a richer description of the distribution of values. Additionally, the quartile and `whikser` values from the `boxplot` are shown inside the violin. The downside is that, because the `violinplot` uses a KDE, there are some other parameters that may need tweaking, adding some complexity relative to the straightforward `boxplot`. It can also be useful to combine `swarmplot()` or `striplot()` with a box plot or violin plot to show each observation along with a summary of the distribution:

```python
g = sns.catplot(x="day", y="total_bill", kind="violin", inner=None, data=tips)
sns.swarmplot(x="day", y="total_bill", color="k", size=3, data=tips, ax=g.ax);
```

##### Statistical estimation within categories

For other applications, rather than showing the distribution within each category, you might want to show an estimate of the central tendency of the values. A familiar style of plot that accomplishes this goal is a bar plot. In `seaborn`, the `barplot()` function operates on a full dataset and applies a function to obtain the estimate (taking the mean by default). When there are multiple observations in each category, it also uses bootstrapping to compute a confidence interval around the estimate and plots that using error bars.
A special case for the bar plot is when you want to show the number of observations in each category rather than computing a statistic for a second variable. This is similar to a histogram over a categorical, rather than quantitative, variable. In `seaborn`, it is easy to do so with the `countplot()` function. An alternative style for visualizing the same information is offered by the `pointplot()` function. This function also encodes the value of the estimate with height on the other axis, but rather than showing a full bar, it plots the point estimate and confidence interval. Additionally, `pointplot()` connects points from the same hue category. This makes it easy to see how the main relationship is changing as a function of the hue semantic, because your eyes are quite good at picking up on differences of slopes:

#### Visualizing the distribution of a dataset

##### Plotting univariate distributions

The most convenient way to take a quick look at a univariate distribution in `seaborn` is the `distplot()` function. By default, this will draw a histogram and fit a kernel density estimate. Histograms are likely familiar, and a hist function already exists in `matplotlib`. A histogram represents the distribution of data by forming bins along the range of the data and then drawing bars to show the number of observations that fall in each bin. To illustrate this, let’s remove the density curve and add a rug plot, which draws a small vertical tick at each observation. You can make the rug plot itself with the `rugplot()` function, but it is also available in `distplot()`:

```python
sns.distplot(x, kde=False, rug=True);
```

Drawing a KDE is more computationally involved than drawing a histogram. 

##### Plotting bivariate distributions

It can also be useful to visualize a bivariate distribution of two variables. The easiest way to do this in `seaborn` is to just use the `jointplot()` function, which creates a multi-panel figure that shows both the bivariate (or joint) relationship between two variables along with the univariate (or marginal) distribution of each on separate axes. The most familiar way to visualize a bivariate distribution is a `scatterplot`, where each observation is shown with point at the x and y values. This is analogous to a rug plot on two dimensions. You can draw a `scatterplot` with the `matplotlib plt.scatter` function, and it is also the default kind of plot shown by the `jointplot()` function. The bivariate analogue of a histogram is known as a `“hexbin”` plot, because it shows the counts of observations that fall within hexagonal bins. This plot works best with relatively large datasets. It’s available through the `matplotlib plt.hexbin` function and as a style in `jointplot()`. It looks best with a white background:

The `jointplot()` function uses a `JointGrid` to manage the figure. For more flexibility, you may want to draw your figure by using `JointGrid` directly. `jointplot()` returns the `JointGrid` object after plotting, which you can use to add more layers or to tweak other aspects of the visualization:

##### Visualizing pairwise relationships in a dataset

To plot multiple pairwise bivariate distributions in a dataset, you can use the `pairplot()` function. This creates a matrix of axes and shows the relationship for each pair of columns in a `DataFrame`. by default, it also draws the univariate distribution of each variable on the diagonal Axes. Much like the relationship between `jointplot()` and `JointGrid`, the `pairplot()` function is built on top of a `PairGrid` object, which can be used directly for more flexibility.

#### Building structured multi-plot grids

##### Conditional small multiples

The `FacetGrid` class is useful when you want to visualize the distribution of a variable or the relationship between multiple variables separately within subsets of your dataset. A `FacetGrid` can be drawn with up to three dimensions: `row, col, and hue`. The first two have obvious correspondence with the resulting array of axes; think of the hue variable as a third dimension along a depth axis, where different levels are plotted with different colors.

The class is used by initializing a `FacetGrid` object with a `dataframe` and the names of the variables that will form the row, column, or hue dimensions of the grid. These variables should be categorical or discrete, and then the data at each level of the variable will be used for a facet along that axis. 

The main approach for visualizing data on this grid is with the `FacetGrid.map()` method. Provide it with a plotting function and the name(s) of variable(s) in the `dataframe` to plot. 

The size of the figure is set by providing the height of each facet, along with the aspect ratio:

Once you’ve drawn a plot using `FacetGrid.map()`, you may want to adjust some aspects of the plot. There are also a number of methods on the `FacetGrid` object for manipulating the figure at a higher level of abstraction. The most general is `FacetGrid.set()`, and there are other more specialized methods like `FacetGrid.set_axis_labels()`, which respects the fact that interior facets do not have axis labels. 

For even more customization, you can work directly with the underling `matplotlib` Figure and Axes objects, which are stored as member attributes at `fig` and `axes` (a two-dimensional array), respectively. When making a figure without row or column faceting, you can also use the ax attribute to directly access the single axes.

You’re not limited to existing `matplotlib` and `seaborn` functions when using `FacetGrid`. However, to work properly, any function you use must follow a few rules:

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

##### plotting pairwise data relationships

`PairGrid` also allows you to quickly draw a grid of small subplots using the same plot type to visualize data in each. In a `PairGrid`, each row and column is assigned to a different variable, so the resulting plot shows each pairwise relationship in the dataset. This style of plot is sometimes called a `“scatterplot matrix”`, as this is the most common way to show each relationship, but `PairGrid` is not limited to `scatterplots`. It is important to understand the differences between a `FacetGrid` and a `PairGrid`. In the former, each facet shows the same relationship conditioned on different levels of other variables. In the latter, each plot shows a different relationship (although the upper and lower triangles will have mirrored plots). Using `PairGrid` can give you a very quick, very high-level summary of interesting relationships in your dataset. The basic usage of the class is very similar to `FacetGrid`. 

#### Matrix plots

```python
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(uniform_data)
'''fmt : string, optional,String formatting code to use when adding annotations.'''
```

#### Controlling figure aesthetics

`Seaborn` splits `matplotlib` parameters into two independent groups. The first group sets the aesthetic style of the plot, and the second scales various elements of the figure so that it can be easily incorporated into different contexts. The interface for manipulating these parameters are two pairs of functions. To control the style, use the `axes_style()` and `set_style()` functions. To scale the plot, use the `plotting_context()` and `set_context()` functions. In both cases, the first function returns a dictionary of parameters and the second sets the `matplotlib` defaults.

##### `Seaborn` figure styles

There are five preset `seaborn` themes: `darkgrid, whitegrid, dark, white, and ticks`. The default theme is `darkgrid`. Both the `white` and `ticks` styles can benefit from removing the top and right axes spines, which are not needed. The `seaborn` function `despine()` can be called to remove them. Some plots benefit from offsetting the spines away from the data, which can also be done when calling `despine()`. When the ticks do not cover the whole range of the axis, the `trim` parameter will limit the range of the surviving spines.
Although it’s easy to switch back and forth, you can also use the `axes_style()` function in a with statement to temporarily set plot parameters. This also allows you to make figures with differently-styled axes. If you want to customize the `seaborn` styles, you can pass a dictionary of parameters to the `rc` argument of `axes_style()` and `set_style()`. Note that you can only override the parameters that are part of the style definition through this method. (However, the higher-level `set()` function takes a dictionary of any `matplotlib parameters`). If you want to see what parameters are included, you can just call the function with no arguments, which will return the current settings

##### Scaling plot elements

A separate set of parameters control the scale of plot elements, which should let you use the same code to make plots that are suited for use in settings where larger or smaller plots are appropriate. The four preset contexts, in order of relative size, are `paper, notebook, talk`, and `poster`. The notebook style is the default, and was used in the plots above. Most of what you now know about the style functions should transfer to the context functions. You can call `set_context()` with one of these names to set the parameters, and you can override the parameters by providing a dictionary of parameter values. You can also independently scale the size of the font elements when changing the context. (This option is also available through the top-level `set()` function).
Similarly, you can temporarily control the scale of figures nested under a with statement. Both the style and the context can be quickly configured with the `set()` function.

#### Choosing color palettes

The most important function for working with discrete color palettes is `color_palette()`. This function provides an interface to many (though not all) of the possible ways you can generate colors in `seaborn`, and it’s used internally by any function that has a palette argument. `color_palette()` will accept the name of any `seaborn` palette or `matplotlib colormap` (except jet, which you should never use). It can also take a list of colors specified in any valid `matplotlib` format (`RGB` tuples, hex color codes, or HTML color names). The return value is always a list of `RGB` tuples. Finally, calling `color_palette()` with no arguments will return the current default color cycle. A corresponding function, `set_palette()`, takes the same arguments and will set the default color cycle for all plots. You can also use `color_palette()` in a with statement to temporarily change the default palette. It is generally not possible to know what kind of color palette or color-map is best for a set of data without knowing about the characteristics of the data. Following that, we’ll break up the different ways to use `color_palette()` and other `seaborn` palette functions by the three general kinds of color palettes: qualitative, sequential, and diverging.

##### Qualitative color palettes

Qualitative palettes are best when you want to distinguish discrete chunks of data that do not have an inherent ordering. When importing `seaborn`, the default color cycle is changed to a set of six colors that evoke the standard `matplotlib` color cycle while aiming to be a bit more pleasing to look at. There are six variations of the default theme, called `deep, muted, pastel, bright, dark`, and `colorblind`. When you have an arbitrary number of categories to distinguish without emphasizing any one, the easiest approach is to draw evenly-spaced colors in a circular color space. This is what most `seaborn` functions default to when they need to use more colors than are currently set in the default color cycle.

The most common way to do this uses the `hls` color space, which is a simple transformation of `RGB` values. There is also the `hls_palette()` function that lets you control the lightness and saturation of the colors. `seaborn` provides an interface to the `husl` system, which also makes it easy to select evenly spaced hues while keeping the apparent brightness and saturation much more uniform. here is similarly a function called `husl_palette()` that provides a more flexible interface to this system.

##### Sequential color palettes

The second major class of color palettes is called “sequential”. This kind of color mapping is appropriate when data range from relatively low or uninteresting values to relatively high or interesting values. Although there are cases where you will want discrete colors in a sequential palette, it is more common to use them as a `colormap` in functions like `kdeplot()` and `heatmap()`.

Like in `matplotlib`, if you want the lightness ramp to be reversed, you can add a `_r` suffix to the palette name. The `cubehelix` color palette system makes sequential palettes with a linear increase or decrease in brightness and some variation in hue. This means that the information in your `colormap` will be preserved when converted to black and white when viewed by a colorblind individual.

For a simpler interface to custom sequential palettes, you can use `light_palette()` or `dark_palette()`, which are both seeded with a single color and produce a palette that ramps either from light or dark desaturated values to that color. 

```python
sns.palplot(sns.light_palette("green"))

sns.palplot(sns.light_palette((210, 90, 60), input="husl"))
sns.palplot(sns.dark_palette("muted purple", input="xkcd"))
```

By default, the input can be any valid `matplotlib` color. Alternate interpretations are controlled by the input argument. Currently you can provide tuples in `hls` or `husl` space along with the default `rgb`, and you can also seed the palette with any valid `xkcd` color.

##### Diverging color palettes

The third class of color palettes is called “diverging”. These are used for data where both large low and high values are interesting. There is also usually a well-defined midpoint in the data. The rules for choosing good diverging palettes are similar to good sequential palettes, except now you want to have two relatively subtle hue shifts from distinct starting hues that meet in an under-emphasized color at the midpoint

You can also use the `seaborn` function `diverging_palette()` to create a custom `colormap` for diverging data. This function makes diverging palettes using the `husl` color system. You pass it two hues and, optionally, the lightness and saturation values for the extremes. Using `husl` means that the extreme values, and the resulting ramps to the midpoint, will be well-balanced.

```python
sns.palplot(sns.diverging_palette(220, 20, n=7))
sns.palplot(sns.diverging_palette(145, 280, s=85, l=25, n=7))
```