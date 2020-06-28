#### Plotting with Basic Glyphs

Note that `Bokeh` plots created using the `bokeh.plotting` interface come with a default set of tools, and default visual styles.

##### Scatter Markers

To scatter circle markers on a plot, use the `circle()` method of `Figure`. Similarly, to scatter square markers, use the `square()` method of `Figure`. All the markers have the same set of properties: `x, y`, size, and angle. Additionally, `circle()` has a radius property that can be used to specify data-space units.

```python
p = figure(plot_width=400, plot_height=400)
p.square([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=20, color="olive", alpha=0.5)
'''asterisk(), circle(), circle_cross(), circle_x(), cross(), dash(), diamond(), diamond_cross(), inverted_triangle(), square(), square_cross(), square_x(), triangle(), x()'''
```

##### Line Glyphs

generate a single line glyph from one dimensional sequences of x and y points using the `line()` glyph method. 

For some kinds of data, it may be more appropriate to draw discrete steps between data points, instead of connecting points with linear segments. The `step()` glyph method can be used to accomplish this: Step levels can be drawn before, after, or centered on the x-coordinates, as configured by the mode parameter. 

Sometimes it is useful to plot multiple lines all at once. This can be accomplished with the `multi_line()` glyph method. `NaN` values can be passed to `line()` and `multi_line()` glyphs. In this case, you end up with single logical line objects, that have multiple disjoint components when rendered.

```python
p.line([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], line_width=2)
p.step([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], line_width=2, mode="center")
p.multi_line([[1, 3, 2], [3, 4, 6, 6]], [[2, 1, 4], [4, 7, 8, 5]],
             color=["firebrick", "navy"], alpha=[0.8, 0.3], line_width=4)
source = ColumnDataSource(data=dict(x=[1, 2, 3, 4, 5],
    y1=[1, 2, 4, 3, 4],y2=[1, 4, 2, 2, 3],))
p = figure(plot_width=400, plot_height=400)
p.vline_stack(['y1', 'y2'], x='x', source=source)
```

In some instances, it is desirable to stack lines that are aligned on a common index. The `vline_stack()` and `hline_stack()` convenience methods can be used to accomplish this. Note the these methods stack columns from an explicit supplied `ColumnDataSource`。

##### Bars and Rectangles

To draw axis aligned rectangles by specifying the **left, right, top, and bottom** positions, use the `quad()` glyph function. To draw arbitrary rectangles by specifying a **center point, width, height, and angle**, use the `rect()` glyph function. 
When drawing rectangular bars it is often more convenient to have coordinates that are a hybrid of the two systems above. `Bokeh` provides the `hbar()` and `vbar()` glyphs function for this purpose. To draw vertical bars by specifying **a (center) x-coordinate, width, and top and bottom** endpoints, use the `vbar()` glyph function. To draw horizontal bars by specifying **a (center) y-coordinate, height, and left and right** endpoints, use the `hbar()` glyph function:

```python
p.vbar(x=[1, 2, 3], width=0.5, bottom=0,top=[1.2, 2.5, 3.7], color="firebrick")
p.hbar(y=[1, 2, 3], height=0.5, left=0,right=[1.2, 2.5, 3.7], color="navy")

source = ColumnDataSource(data=dict( y=[1, 2, 3, 4, 5],x1=[1, 2, 4, 3, 4],
    x2=[1, 4, 2, 2, 3],))
p.hbar_stack(['x1', 'x2'], y='y', height=0.8, color=("grey", "lightgrey"), source=source)
p.quad(top=[2, 3, 4], bottom=[1, 2, 3], left=[1, 2, 3], right=[1.2, 2.5, 3.7], color="#B3DE69")
p.rect(x=[1, 2, 3], y=[1, 2, 3], width=0.2, height=40, color="#CAB2D6",angle=pi/3, height_units="screen")
```

##### Hex Tiles

`Bokeh` can plot hexagonal tiles, which are often used for showing binned aggregations. The `hex_tile()` method takes a **size** parameter to define the size of the hex grid, and **axial coordinates** to specify which tiles are present.

```python
from bokeh.util.hex import axial_to_cartesian
q = np.array([0,  0, 0, -1, -1,  1, 1])
r = np.array([0, -1, 1,  0,  1, -1, 0])
p.hex_tile(q, r, size=1, fill_color=["firebrick"]*3 + ["navy"]*4, line_color="white", alpha=0.5)
x, y = axial_to_cartesian(q, r, 1, "pointytop")

p.text(x, y, text=["(%d, %d)" % (q,r) for (q, r) in zip(q, r)],
       text_baseline="middle", text_align="center")
```

##### Directed Areas

Directed areas are filled regions between two series that share a common index. For instance, a vertical directed area has one x coordinate array, and two y coordinate arrays, $y_1$ and $y_2$, which will be filled between.

###### Single Areas

A single directed area between two aligned series can be created in the vertical direction with `varea()` or in the horizontal direction with `harea()`.

```python
p.varea(x=[1, 2, 3, 4, 5],y1=[2, 6, 4, 3, 5],y2=[1, 4, 2, 2, 3])
```

###### Stacked Areas

It is often desirable to stack directed areas. This can be accomplished with the `varea_stack()` and `harea_stack()` convenience methods. Note the these methods stack columns from an explicit supplied `ColumnDataSource`

```python
source = ColumnDataSource(data=dict(x=[1, 2, 3, 4, 5],y1=[1, 2, 4, 3, 4],
    y2=[1, 4, 2, 2, 3],))
p.varea_stack(['y1', 'y2'], x='x', color=("grey", "lightgrey"), source=source)
```

##### Patches and Polygons

###### Patches

generate a single polygonal patch glyph from one dimensional sequences of x and y points using the `patch()` glyph method. Sometimes it is useful to plot multiple polygonal patches all at once. This can be accomplished with the `patches()` glyph method. Just as with `line()` and `multi_line()`, `NaN` values can be passed to `patch()` and `patches()` glyphs. In this case, you end up with single logical patch objects, that have multiple disjoint components when rendered

```python
p.patch([1, 2, 3, 4, 5], [6, 7, 8, 7, 3], alpha=0.5, line_width=2)
p.patches([[1, 3, 2], [3, 4, 6, 6]], [[2, 1, 4], [4, 7, 8, 5]],
          color=["firebrick", "navy"], alpha=[0.8, 0.3], line_width=2)
p.patch([1, 2, 3, nan, 4, 5, 6], [6, 7, 5, nan, 7, 3,6],alpha=0.5,line_width=2)
```

###### Polygons

`p.multi_polygons(xs=[[[[1, 1, 2, 2]]]], ys=[[[[3, 4, 4, 3]]]])`产生一个`polygon`需要一维数组， 在`polygon`中生成`hole`需要二维数组，一组`MultiPolygon`中含有多个`polygon`需要三位数组，产生多个`MultiPolygon`需要四维数组。所以输入的位置参数为一个四维数组。

The `multi_polygons()` glyph uses nesting to accept a variety of information relevant to polygons. Anything that can be rendered as a `patches()` can also be rendered as `multi_polygons()`, but additionally `multi_polygons()` can render holes inside each polygon. Sometimes one conceptual polygon is composed of multiple polygon geometries. The top level of nesting is used to separate each `MultiPolygon` from the others. Each `MultiPolygon` can be thought of as a row in the data source - potentially with a corresponding label or color. 

```python
p.multi_polygons(xs=[[[[1, 1, 2, 2]]]],ys=[[[[3, 4, 4, 3]]]])
p.multi_polygons(xs=[[[ [1, 2, 2, 1], [1.2, 1.6, 1.6], [1.8, 1.8, 1.6] ]]],
                 ys=[[[ [3, 3, 4, 4], [3.2, 3.6, 3.2], [3.4, 3.8, 3.8] ]]])
p.multi_polygons(xs=[[[ [1, 1, 2, 2], [1.2, 1.6, 1.6], [1.8, 1.8, 1.6] ], [ [3, 4, 3] ]]],
                 ys=[[[ [4, 3, 3, 4], [3.2, 3.2, 3.6], [3.4, 3.8, 3.8] ], [ [1, 1, 3] ]]])
p.multi_polygons(
    xs=[
        [[ [1, 1, 2, 2], [1.2, 1.6, 1.6], [1.8, 1.8, 1.6] ], [ [3, 3, 4] ]],
        [[ [1, 2, 2, 1], [1.3, 1.3, 1.7, 1.7] ]]],
    ys=[
        [[ [4, 3, 3, 4], [3.2, 3.2, 3.6], [3.4, 3.8, 3.8] ], [ [1, 3, 1] ]],
        [[ [1, 1, 2, 2], [1.3, 1.7, 1.7, 1.3] ]]],
    color=['blue', 'red'])
```

The `oval()` glyph method accepts the same properties as `rect()`, but renders oval shapes. The `ellipse()` glyph accepts the same properties as `oval()` and `rect()` but renders ellipse shapes, which are different from oval ones. In particular, the same value for width and height will render a circle using the `ellipse()` glyph but not the `oval()` one.

```python
p.oval(x=[1, 2, 3], y=[1, 2, 3], width=0.2, height=40, color="#CAB2D6",
       angle=pi/3, height_units="screen")
p.ellipse(x=[1, 2, 3], y=[1, 2, 3], width=[0.2, 0.3, 0.1], height=0.3,
          angle=pi/3, color="#CAB2D6")
```

##### Segments and Rays

Sometimes it is useful to be able to draw many individual line segments at once. `Bokeh` provides the `segment()` and `ray()` glyph methods to render these. The `segment()` function accepts start points $x_0, y_0$ and end points $x_1, y1$ and renders segments between these: The `ray()` function accepts start points $x, y$ with a length and an **angle**. The default angle_units are "rad" but can also be changed to "deg". 

```python
p.segment(x0=[1, 2, 3], y0=[1, 2, 3], x1=[1.2, 2.4, 3.1],
          y1=[1.2, 2.5, 3.7], color="#F4A582", line_width=3)
p.ray(x=[1, 2, 3], y=[1, 2, 3], length=45, angle=[30, 45, 60],
      angle_units="deg", color="#FB8072", line_width=2)
```

##### Wedges and Arcs

To draw a simple line `arc`, `Bokeh` provides the `arc()` glyph method, which accepts **radius, start_angle, and end_angle** to determine position. Additionally, the **direction** property determines whether to render clockwise or anti-clockwise between the start and end angles. The `wedge()` glyph method accepts the same properties as `arc()`, but renders a filled wedge instead: The `annular_wedge()` glyph method is similar to `arc()`, but draws a filled area. It accepts a **inner_radius and outer_radius** instead of just `radius` Finally, the `annulus()` glyph methods, which accepts **inner_radius and outer_radius**, can be used to draw filled rings. 

```python
p.arc(x=[1, 2, 3], y=[1, 2, 3], radius=0.1, start_angle=0.4, end_angle=4.8, color="navy")
p.wedge(x=[1, 2, 3], y=[1, 2, 3], radius=0.2, start_angle=0.4, end_angle=4.8,
        color="firebrick", alpha=0.6, direction="clock")
p.annular_wedge(x=[1, 2, 3], y=[1, 2, 3], inner_radius=0.1, outer_radius=0.25,
                start_angle=0.4, end_angle=4.8, color="green", alpha=0.6)
p.annulus(x=[1, 2, 3], y=[1, 2, 3], inner_radius=0.1, outer_radius=0.25,
          color="orange", alpha=0.6)
```

##### Specialized Curves

###### Combining Multiple Glyphs

Combining multiple glyphs on a single plot is a matter of calling more than one glyph method on a single Figure.

```python
x = [1, 2, 3, 4, 5]
y = [6, 7, 8, 7, 3]
p = figure(plot_width=400, plot_height=400)
p.line(x, y, line_width=2)
p.circle(x, y, fill_color="white", size=8)
```

##### Setting Ranges

 By default, `Bokeh` will attempt to automatically set the data bounds of plots to fit snugly around the data. Sometimes you may need to set a `plot’s` range explicitly. This can be accomplished by setting the **x_range or y_range** properties using a `Range1d` object that gives the start and end points of the range you want. As a convenience, the `figure()` function can also accept tuples of (start, end) as values for the **x_range or y_range** parameters. 

```python
from bokeh.models import Range1d
p.y_range = Range1d(0, 15)
```

##### Specifying Axis Types

###### Categorical Axes

Categorical axes are created by specifying a `FactorRange` for one of the plot ranges (or a lists of factors to be converted to one).  

```python
factors = ["a", "b", "c", "d", "e", "f", "g", "h"]
x = [50, 40, 65, 10, 25, 37, 80, 60]

p = figure(y_range=factors)
p.circle(x, factors, size=15, fill_color="orange", line_color="green", line_width=3)
```

###### $\text{Datetime}$ Axes

When dealing with time series data, or any data that involves dates or times, it is desirable to have an axis that can display labels that are appropriate to different date and time scales. We have seen how to use the `figure()` function to create plots using the `bokeh.plotting` interface. This function accepts **x_axis_type and y_axis_type** as arguments. To specify a `datetime` axis, pass `"datetime"` for the value of either of these parameters. 

```python
df = pd.DataFrame(AAPL)
df['date'] = pd.to_datetime(df['date'])

p = figure(plot_width=800, plot_height=250, x_axis_type="datetime")
p.line(df['date'], df['close'], color='navy', alpha=0.5)
```

###### Log Scale Axes

When dealing with data that grows exponentially or is of many orders of magnitude, it is often necessary to have one axis on a log scale. Another scenario involves plotting data that has a power law relationship, when it is desirable to use log scales on both axes. As we saw above, the `figure()` function accepts **x_axis_type and y_axis_type** as arguments. To specify a log axis, pass` "log"` for the value of either of these parameters. 

```python
x = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
y = [10**xx for xx in x]

p = figure(plot_width=400, plot_height=400, y_axis_type="log")
p.line(x, y, line_width=2)
p.circle(x, y, fill_color="white", size=8)
```

###### Twin Axes

It is possible to add multiple axes representing different ranges to a single plot. To do this, configure the plot with `“extra”` named ranges in the **extra_x_range and extra_y_range** properties. Then these named ranges can be referred to when adding new glyph methods, and also to add new axes objects using the add_layout method on Plot. 

```python
from bokeh.models import LinearAxis, Range1d
x = arange(-2*pi, 2*pi, 0.1)
y = sin(x)
y2 = linspace(0, 100, len(y))

p = figure(x_range=(-6.5, 6.5), y_range=(-1.1, 1.1))
p.circle(x, y, color="red")

p.extra_y_ranges = {"foo": Range1d(start=0, end=100)}
p.circle(x, y2, color="blue", y_range_name="foo")
p.add_layout(LinearAxis(y_range_name="foo"), 'left')
```

#### Providing Data for Plots and Tables

##### `ColumnDataSource`

The `ColumnDataSource` is the core of most `Bokeh` plots, providing the data that is visualized by the glyphs of the plot. With the `ColumnDataSource`, it is easy to share data between multiple plots and widgets. When the same `ColumnDataSource` is used to drive multiple `renderers`, selections of the data source are also shared. Thus it is possible to use a select tool to choose data points from one plot and have them automatically highlighted in a second plot.

At the most basic level, a `ColumnDataSource` is simply a mapping between column names and lists of data. The `ColumnDataSource` takes a data parameter which is a `dict`, with string column names as keys and lists (or arrays) of data values as values. If one positional argument is passed in to the `ColumnDataSource` initializer, it will be taken as data. Once the `ColumnDataSource` has been created, it can be passed into the `source` parameter of plotting methods which allows you to pass a column’s name as a stand in for the data values.

###### Pandas

If a `DataFrame` is used, the `CDS` will have columns corresponding to the columns of the `DataFrame`. The index of the `DataFrame` will be reset, so if the `DataFrame` has a named index column, then `CDS` will also have a column with this name. However, if the index name is `None`, then the `CDS` will be assigned a generic name. It will be `index` if it is available, and `level_0` otherwise.

###### Pandas `MultiIndex`

All `MultiIndex` columns and `indices` will be flattened before forming the `ColumnsDataSource`. For the index, **an index of tuples** will be created, and the names of the `MultiIndex` joined with an **underscore**. The column names will also be joined with an underscore. 

###### Pandas `GroupBy`

If a `GroupBy` object is used, the `CDS` will have columns corresponding to the result of calling `group.describe()`. The describe method generates columns for statistical measures for all the non-grouped original columns. The resulting `DataFrame` has `MultiIndex` columns with the original column name and the computed measure, so it will be flattened using the aforementioned scheme. 

###### Streaming

`ColumnDataSource` streaming is an efficient way to append new data to a `CDS`. By using the stream method, `Bokeh` only sends new data to the browser instead of the entire dataset. The stream method takes a `new_data` parameter containing a `dict` mapping column names to sequences of data to be appended to the respective columns. It additionally takes an optional argument `rollover`, which is the maximum length of data to keep. The default rollover value of None allows data to grow unbounded.

###### Patching

`ColumnDataSource` patching is an efficient way to update slices of a data source. By using the patch method, `Bokeh` only needs to send new data to the browser instead of the entire dataset. The patch method should be passed a `dict` mapping column names to list of tuples that represent a patch change to apply.

```python
(index, new_value)  # replace a single column value
(slice, new_values) # replace several column values
```

##### Transforming Data

We have seen above how data can be added to a `ColumnDataSource` to drive `Bokeh` plots. This can include raw data or data that we explicitly transform ourselves, for example a column of colors created to control how the Markers in a scatter plot should be shaded. It is also possible to specify transforms that only occur in the browser. This can be useful to reduce both code as well as the amount of data that has to be sent into the browser.

###### Colors

To perform linear color mapping in the browser, the `linear_cmap()` function may be used. It accepts the name of a `ColumnDataSource` column to `colormap`, a palette (which can be a built-in palette name, or an actual list of colors), and min/max values for the color mapping range. The result can be passed to a **color** property on glyphs:

###### Markers

It is also possible to map categorical data to marker types. the use of `factor_mark()` to display different markers or different categories in the input data. the use of `factor_cmap()` to `colormap` those same categories

##### Filtering Data

It is often desirable to focus in on a portion of data that has been `subsampled` or filtered from a larger dataset. `Bokeh` allows you to specify a view of a data source that represents a subset of data. By having a view of the data source, the underlying data does not need to be changed and can be shared across plots. The view consists of one or more filters that select the rows of the data source that should be bound to a specific glyph. To plot with a subset of data, you can create a `CDSView` and pass it in as a **view** argument to the renderer-adding methods on the Figure. The `CDSView` has two properties, **source and filters**. source is the `ColumnDataSource` that the view is associated with. filters is a list of Filter objects.

```python
from bokeh.models import ColumnDataSource, CDSView
source = ColumnDataSource(some_data)
view = CDSView(source=source, filters=[filter1, filter2])
p = figure()
p.circle(x="x", y="y", source=source, view=view)
```

###### `IndexFilter`

The `IndexFilter` is the simplest filter type. It has an `indices` property which is a list of integers that are the `indices` of the data you want to be included in the plot.

```python
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, CDSView, IndexFilter
from bokeh.plotting import figure, show

source = ColumnDataSource(data=dict(x=[1, 2, 3, 4, 5], y=[1, 2, 3, 4, 5]))
view = CDSView(source=source, filters=[IndexFilter([0, 2, 4])])

p = figure(plot_height=300, plot_width=300)
p.circle(x="x", y="y", size=10, hover_color="red", source=source)

p_filtered = figure(plot_height=300, plot_width=300, tools=tools)
p_filtered.circle(x="x", y="y", size=10, hover_color="red", source=source, view=view)

show(gridplot([[p, p_filtered]]))
```

###### `BooleanFilter`

A `BooleanFilter` selects rows from a data source through a list of True or False values in its booleans property.

```python
source = ColumnDataSource(data=dict(x=[1, 2, 3, 4, 5], y=[1, 2, 3, 4, 5]))
booleans = [True if y_val > 2 else False for y_val in source.data['y']]
view = CDSView(source=source, filters=[BooleanFilter(booleans)])
```

###### `GroupFilter`

The `GroupFilter` allows you to select rows from a dataset that have a specific value for a categorical variable. The `GroupFilter` has two properties, **column_name**, the name of column in the `ColumnDataSource`, and **group**, the value of the column to select for.

```python
source = ColumnDataSource(flowers)
view1 = CDSView(source=source, filters=[GroupFilter(column_name='species', group='versicolor')])
```

##### Linked Selection

With the ability to specify a subset of data to be used for each glyph renderer, it is easy to share data between plots even when the plots use different subsets of data. By using the same `ColumnDataSource`, selections and hovered inspections of that data source are automatically shared. Selections in either plot are automatically reflected in the other. And hovering on a point in one plot will highlight the corresponding point in the other plot if it exists.

#### Laying out Plots and Widgets

At the heart of the layouts are three core objects `Row, Column`. While you can use these models directly, we recommend using the layout functions `row(), column()`. 

There are two things to keep in mind for best results using layout: All items must have the same sizing mode; Widgets should be inside a widget box.

Consistent sizing mode: Every item in a layout must have the same sizing mode for the layout to behave as expected. It is for this reason that we recommend using the layout functions as they help ensure that all the children of the row or column have the same sizing mode. We hope to lift this restriction in future releases.

Widget boxes: Widgets are HTML objects like buttons, and `dropdown` menus. They behave slightly differently to plots and and putting them in a `widgetbox` is necessary so that they can all work together. In fact, if you try and put a Widget in Row or Column it will be automatically put into a `WidgetBox`. As a result, it is a good idea to wrap your own widgets in a `WidgetBox` using as then you can be sure about how your widgets are getting arranged.

```python
#To display plots or widgets in a vertical fashion, use the column() function
show(column(s1, s2, s3))
#To display plots horizontally, use the row() function.
show(row(s1, s2, s3))
#Layout a group of widgets with the function.
show(column(button_1, slider, button_group, select, button_2, width=300))
show(gridplot([[p1, p2], [None, p3]]))
l = layout([[bollinger],
  [sliders, plot],
  [p1, p2, p3],], sizing_mode='stretch_both')
```

The `gridplot()` function can be used to arrange `Bokeh` Plots in grid layout. `gridplot()` also collects all tools into a single `toolbar`, and the currently active tool is the same for all plots in the grid. It is possible to leave “empty” spaces in the grid by passing None instead of a plot object. The `layout()` function can be used to arrange both Plots and Widgets in a grid, generating the necessary `row()` and `column()` layouts automatically. This allows for quickly spelling a layout.

#### Handling Categorical Data

##### Bars

Since `Bokeh` displays bars in the order the factors are given for the range, “sorting” bars in a bar plot is identical to sorting the factors for the range.
Often times we may want to have bars that are shaded some color. This can be accomplished in different ways. One way is to supply all the colors up front. This can be done by putting all the data, including the colors for each bar, in a `ColumnDataSource`. Then the name of the column containing the colors is passed to figure as the color arguments. 

###### color

Often times we may want to have bars that are shaded some color. This can be accomplished in different ways. One way is to supply all the colors up front. This can be done by putting all the data, including the colors for each bar, in a `ColumnDataSource`. Then the name of the column containing the colors is passed to `vbar` as the `color` (or `line_color`/`fill_color`) arguments.

```python
from bokeh.palettes import Spectral6
fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
counts = [5, 3, 4, 2, 4, 6]
source = ColumnDataSource(data=dict(fruits=fruits, counts=counts, color=Spectral6))
```

Another way to shade the bars is to use a `CategoricalColorMapper` that `colormaps` the bars inside the browser. There is a function `factor_cmap()` that makes this simple to do

```python
source = ColumnDataSource(data=dict(fruits=fruits, counts=counts))

p = figure(x_range=fruits, plot_height=250, toolbar_location=None, title="Fruit Counts")
p.vbar(x='fruits', top='counts', width=0.9, source=source, legend="fruits",
       line_color='white', fill_color=factor_cmap('fruits', palette=Spectral6, factors=fruits))
```

###### Stacked

Another common operation or bar charts is to stack bars on top of one another. `Bokeh` makes this easy to do with the specialized `hbar_stack()` and `vbar_stack()` functions. 

```python
fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
years = ["2015", "2016", "2017"]
exports = {'fruits' : fruits,
           '2015'   : [2, 1, 4, 3, 2, 4],
           '2016'   : [5, 3, 4, 2, 4, 6],
           '2017'   : [3, 2, 4, 4, 5, 3]}
imports = {'fruits' : fruits,
           '2015'   : [-1, 0, -1, -3, -2, -1],
           '2016'   : [-2, -1, -3, -1, -2, -2],
           '2017'   : [-1, -2, -1, 0, -2, -2]}

p.hbar_stack(years, y='fruits', height=0.9, color=GnBu3, source=ColumnDataSource(exports))

p.hbar_stack(years, y='fruits', height=0.9, color=OrRd3, source=ColumnDataSource(imports))
```

For stacked bar plots, `Bokeh` provides some special hover variables that are useful for common cases. When stacking bars, `Bokeh` automatically sets the `name` property for each layer in the stack to be the value of the stack column for that layer. This name value is accessible to hover tools via the `$name` special variable. Additionally, the hover variable `@$name` can be used to look up values from the stack column for each layer. 

##### Grouped

When creating bar charts, it is often desirable to visually display the data according to sub-groups. There are two basic methods that can be used, depending on your use case: using nested categorical coordinates, or applying dodges. 

###### nested categories

If the coordinates of a plot range and data have two or three levels, then `Bokeh` will automatically group the factors on the axis, including a hierarchical tick labeling with separators between the groups. In the case of bar charts, this results in bars grouped together by the top-level factors. This is probably the most common way to achieve grouped bars, especially if you are starting from “tidy” data.

```python
x = [ (fruit, year) for fruit in fruits for year in years ]
counts = sum(zip(data['2015'], data['2016'], data['2017']), ()) # like an hstack

source = ColumnDataSource(data=dict(x=x, counts=counts))

p = figure(x_range=FactorRange(*x), plot_height=250, title="Fruit Counts by Year",
           toolbar_location=None, tools="")

p.vbar(x='x', top='counts', width=0.9, source=source)
```

###### Visual Dodge

Another method for achieving grouped bars is to explicitly specify a visual displacement for the bars. Such a visual offset is also referred to as a dodge. In this scenario, our data is not “tidy”. Instead a single table with rows indexed by factors (fruit, year), we have separate series for each year. We can plot all the year series using separate calls to `vbar` but since every bar in each group has the same fruit factor, the bars would overlap visually. We can prevent this overlap and distinguish the bars visually by using the `dodge()` function to provide an offset for each different call to `vbar`

```python
source = ColumnDataSource(data=data)

p = figure(x_range=fruits, y_range=(0, 10), plot_height=250, title="Fruit Counts by Year",
           toolbar_location=None, tools="")

p.vbar(x=dodge('fruits', -0.25, range=p.x_range), top='2015', width=0.2, source=source,
       color="#c9d9d3", legend=value("2015"))

p.vbar(x=dodge('fruits',  0.0,  range=p.x_range), top='2016', width=0.2, source=source,
       color="#718dbf", legend=value("2016"))

p.vbar(x=dodge('fruits',  0.25, range=p.x_range), top='2017', width=0.2, source=source,
       color="#e84d60", legend=value("2017"))
```

###### Stacked and Grouped

```python
factors = [("Q1", "jan"), ("Q1", "feb"), ("Q1", "mar"),
    ("Q2", "apr"), ("Q2", "may"), ("Q2", "jun"),
    ("Q3", "jul"), ("Q3", "aug"), ("Q3", "sep"),
    ("Q4", "oct"), ("Q4", "nov"), ("Q4", "dec"),]

regions = ['east', 'west']
source = ColumnDataSource(data=dict(x=factors,
    east=[ 5, 5, 6, 5, 5, 4, 5, 6, 7, 8, 6, 9 ],
    west=[ 5, 7, 9, 4, 5, 4, 7, 7, 7, 6, 6, 7 ],))

p = figure(x_range=FactorRange(*factors), plot_height=250, tools="")

p.vbar_stack(regions, x='x', width=0.9, alpha=0.5, color=["blue", "red"], source=source, legend=[value(x) for x in regions])
```

So far we have seen the bar glyphs used to create bar charts, which imply bars drawn from a common baseline. However, the bar glyphs can also be used to represent arbitrary intervals across a range.

##### Scatters

###### Adding Jitter

When plotting many scatter points in a single categorical category, it is common for points to start to visually overlap. In this case, `Bokeh` provides a jitter() function that can automatically apply a random dodge to every point.

```python
DAYS = ['Sun', 'Sat', 'Fri', 'Thu', 'Wed', 'Tue', 'Mon']
source = ColumnDataSource(data)

p = figure(plot_width=800, plot_height=300, y_range=DAYS, x_axis_type='datetime',title="Commits by Time of Day (US/Central) 2012—2016")

p.circle(x='time', y=jitter('day', width=0.6, range=p.y_range),  source=source, alpha=0.3)
```

We’ve seen above how categorical locations can be modified by operations like dodge and jitter. It is also possible to supply an offset to a categorical location explicitly. This is done by adding a numeric value to the end of a category, e.g. ["Jan", 0.2] is the category “Jan” offset by a value of 0.2. For hierachical categories, the value is added at the end of the existing list, e.g. ["West", "Sales", -0,2]. Any numeric value at the end of a list of categories is always interpreted as an offset.

###### heat maps

In all of the cases above, we have had one categorical axis, and one continuous axis. It is possible to have plots with two categorical axes. If we shade the rectangle that defines each pair of categories, we end up with a Categorical `Heatmap`

The plot below shows such a plot, where the x-axis categories are a list of years from 1948 to 2016, and the y-axis categories are the months of the years. Each rectangle corresponding to a (year, month) combination is color mapped by the unemployment rate for that month and year. Since the unemployment rate is a continuous variable, a `LinearColorMapper` is used to `colormap` the plot, and is also passed to a color bar to provide a visual legend on the right:

```python
source = ColumnDataSource(df)
colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
mapper = LinearColorMapper(palette=colors,low=df.rate.min(),high=df.rate.max())

p = figure(plot_width=800, plot_height=300, title="US Unemployment 1948—2016",
           x_range=list(data.index), y_range=list(reversed(data.columns)),
           toolbar_location=None, tools="", x_axis_location="above")

p.rect(x="Year", y="Month", width=1, height=1, source=source,
       line_color=None, fill_color=transform('rate', mapper))
```

#### Configuring Plot Tools

Tools can be grouped into four basic categories: Gestures: These are tools that respond to single gestures, such as a pan movement;  Actions: These are immediate or modal operations that are only activated when their button in the `toolbar` is pressed;  Inspectors: These are passive tools that report information or annotate plots in some way;  Edit Tools: These are sophisticated multi-gesture tools that can add, delete, or modify glyphs on a plot. Since they may respond to several gestures at once, an edit tool will potentially deactivate multiple single-gesture tools at once when it is activated.

By default, `Bokeh` plots come with a `toolbar` above the plot. The `toolbar` location can be specified by passing the` toolbar_location` parameter to the `figure()` function. Valid values are: `"above", "below", "left", "right"`. If you would like to hide the `toolbar` entirely, pass None.

At the lowest `bokeh.models` level, tools are added to a `Plot` by passing instances of `Tool` objects to the `add_tools` method. This explicit way of adding tools works with any `Bokeh` Plot or Plot subclass, such as Figure. Tools can be specified by passing the `tools` parameter to the `figure()` function. The tools parameter accepts a list of tool objects.  Tools can also be supplied conveniently with a comma-separate string containing tool shortcut names. Finally, it is also always possible to add new tools to a plot by passing a tool object to the `add_tools` method of a plot. This can also be done in conjunction with the tools keyword

##### Inspectors

| 函数            | name        | 作用                                                         |
| --------------- | ----------- | ------------------------------------------------------------ |
| `CrosshairTool` | `crosshair` | Th `crosshair` tool draws a `crosshair` annotation over the plot, centered on the current mouse position. |
| `HoverTool`     | hover       | The hover tool is a passive inspector tool.                  |

By default, the hover tool will generate a “tabular” `tooltip` where each row contains a label, and it's associated value. The labels and values are supplied as a list of (label, value) tuples. Field names that begin with `$` are “special fields”. These often correspond to values that are intrinsic to the plot, such as the coordinates of the mouse in data or screen space. These special fields are listed here:

`$index`:	index of selected point in the data source
`$name`:	value of the name property of the hovered glyph renderer
`$x`:	x-coordinate under the cursor in data space
`$y`:	y-coordinate under the cursor in data space
`$sx`:	x-coordinate under the cursor in screen (canvas) space
`$sy`:	y-coordinate under the cursor in screen (canvas) space
`$name`:	The name property of the glyph that is hovered over
`$color`:	colors from a data source
Field names that begin with `@` are associated with columns in a `ColumnDataSource`. For instance the field name "@price" will display values from the "price" column whenever a hover is triggered. If the hover is for the 17th glyph, then the hover `tooltip` will correspondingly display the 17th price value. Note that if a column name contains spaces空格, the it must be supplied by surrounding it in curly braces`{}`. Sometimes it is desirable to allow the name of the column be specified indirectly. The field name `@$name` is distinguished in that it will look up the name field on the hovered glyph renderer, and use that value as the column name. 

configure and use the hover tool by setting the `tooltips` argument to `figure`.

By default, values for fields are displayed in a basic numeric format. However it is possible to control the formatting of values more precisely. Fields can be modified by appending a format specified to the end in curly braces. there are other formatting schemes that can be specified for interpreting format strings: `"numeral"`:	Provides a wide variety of formats for numbers, currency, bytes, times, and percentages. 
`"datetime"`:	Provides formats for date and time values; `"printf"`:	Provides formats similar to C-style `“printf”` type specifiers.

```python
HoverTool(tooltips=[( 'date',   '@date{%F}'),
        ( 'close',  '$@{adj close}{%0.2f}' ),( 'volume', '@volume{0.00 a}'),],
    formatters={'date': 'datetime', # use 'datetime' formatter for 'date' field
'adj close' : 'printf',   # use 'printf' formatter for 'adj close' field
 },## use default 'numeral' formatter for other fields   
    # display a tooltip whenever the cursor is vertically in line with a glyph
    mode='vline')
```

#### Styling Visual Attributes

##### Color

 `Bokeh` offers many of the standard Brewer palettes, which can be imported from the `bokeh.palettes` module. Color Mappers allow you to encode some data sequence into a palette of colors based on the value in that sequence. The mappers are then set as the color attribute on marker objects. `Bokeh` includes several types of mappers to encode colors: `bokeh.transform.factor_cmap`: Maps colors to specific categorical elements; `bokeh.transform.linear_cmap`: Maps a range of numeric values across the available colors from high to low. `bokeh.transform.log_cmap`: Similar to `linear_cma`p but uses a natural log scale to map the colors. These mapper functions return a `DataSpec` property that can be passed to the color attribute of the glyph. The returned `dataspec` includes a `bokeh.transform` which can be accessed to use the mapper in another context such as to create a `ColorBar` 
Colors properties are used in many places in `Bokeh`, to specify the colors to use for lines, fills or text.

```python
x = [1,2,3,4,5,7,8,9,10]
y = [1,2,3,4,5,7,8,9,10]

mapper = linear_cmap(field_name='y', palette=Spectral6 ,low=min(y) ,high=max(y))
source = ColumnDataSource(dict(x=x,y=y))

p = figure(plot_width=300, plot_height=300, title="Linear Color Map Based on Y")
p.circle(x='x', y='y', line_color=mapper,color=mapper, fill_alpha=1, size=12, source=source)
color_bar = ColorBar(color_mapper=mapper['transform'], width=8,  location=(0,0))
p.add_layout(color_bar, 'right')
```

##### Visual Properties

In order to style the visual attributes of `Bokeh` plots, you first must know what the available properties are.  there are three broad groups of properties that show up often.

###### Line Properties

`line_color`: color to use to stroke lines with; `line_width`: line stroke width in units of pixels; `line_alpha`: floating point between 0 (transparent) and 1 (opaque); `line_join`: how path segments should be joined together; `line_cap`: how path segments should be terminated; `line_dash`:a line style to use: `'solid', 'dashed', 'dotted', 'dotdash', 'dashdot'`, an array of integer pixel distances that describe the on-off pattern of dashing to use, a string of spaced integers matching the regular expression that describe the on-off pattern of dashing to use; `line_dash_offset`: the distance in pixels into the line_dash that the pattern should start from

###### Fill Properties

fill_color: color to use to fill paths with
fill_alpha: floating point between 0 (transparent) and 1 (opaque)

###### Text Properties

`text_font`: font name; `text_font_size` font size in `px`; `text_font_style`: font style to use; `text_color`: color to use to render text with; `text_alpha`: floating point between 0 (transparent) and 1 (opaque); `text_align`: horizontal anchor point for text;  `text_baseline` vertical anchor point for text

Glyph `renderers, axes, grids, and annotations` all have a visible property that can be used to turn them on and off.

```python
invisible_line = p.line([1, 2, 3], [2, 1, 2], line_color="pink")
invisible_line.visible = False
p.xaxis.visible = False
```

##### Plots

`Bokeh` plots comprise graphs of objects that represent all the different parts of the plot: grids, axes, glyphs, etc. In order to style `Bokeh` plots, it is necessary to first find the right object, then set its various attributes. Some objects have convenience methods to help find the objects of interest . But there is also a `select()` method on Plot that can be used to query for `Bokeh` plot objects more generally.

Plot objects themselves have many visual characteristics that can be styled: the dimensions of the plot, backgrounds, borders, outlines, etc. 

###### Dimensions

The dimensions (width and height) of a Plot are controlled by `plot_width` and `plot_height` attributes. These values are in screen units, and they control the size of the entire canvas area, including any axes or titles. If you are using the `bokeh.plotting` interface, then these values can be passed to `figure()` as a convenience

###### Title

The styling of the plot title is controlled by the properties of `Title` annotation, which is available as the `.title` property on the Plot. Most of the standard `Text` Properties are available, with the exception of `text_align` and `text_baseline` which do not apply. For positioning the title relative to the entire plot, use the properties **align and offset**.

```python
p = figure(plot_width=400, plot_height=400, title="Some Title")
p.title.text_color = "olive"
p.title.text_font = "times"
p.title.text_font_style = "italic"
```

###### Background

The background fill style is controlled by the `background_fill_color` and `background_fill_alpha` properties of the Plot object

```python
p = figure(plot_width=400, plot_height=400)
p.background_fill_color = "beige"
p.background_fill_alpha = 0.5
```

###### Border

The border fill style is controlled by the `border_fill_color` and `border_fill_alpha` properties of the Plot object. You can also set the minimum border on each side with the properties: `min_border_left, min_border_right, min_border_top, min_border_bottom`. Additionally, setting `min_border` will apply a minimum border setting to all sides as a convenience.

```python
p = figure(plot_width=400, plot_height=400)
p.border_fill_color = "whitesmoke"
p.min_border_left = 80
```

###### Outline

The styling of the outline of the plotting area is controlled by a set of Line Properties on the Plot, that are prefixed with `outline_`.

```python
p = figure(plot_width=400, plot_height=400)
p.outline_line_width = 7
p.outline_line_alpha = 0.3
p.outline_line_color = "navy"
```

###### Glyphs

To style the fill, line, or text properties of a glyph, it is first necessary to obtain a specific `GlyphRenderer`. When using the `bokeh.plotting` interface, the glyph functions return the renderer. Then, the glyph itself is obtained from the `.glyph` attribute of a `GlyphRenderer`. This is the object to set fill, line, or text property values for.

```python
p = figure(plot_width=400, plot_height=400)
r = p.circle([1,2,3,4,5], [2,5,8,2,7])

glyph = r.glyph
glyph.fill_alpha = 0.2
glyph.line_color = "firebrick"
glyph.line_dash = [6, 3]
```

###### Selected and Unselected Glyphs

The styling of selected and non-selected glyphs can be customized by setting the `selection_glyph` and/or `nonselection_glyph` attributes of the `GlyphRenderer` either manually or by passing them to `add_glyph()`. Click or tap circles on the plot to see the effect on the selected and non-selected glyphs. To clear the selection and restore the original state, click anywhere in the plot outside of a circle.
If you just need to set the **color or alpha** parameters of the selected or non-selected glyphs, this can be accomplished even more simply by providing color and alpha arguments to the glyph function, prefixed by `"selection_" or "nonselection_"`. 

```python
renderer = plot.circle([1, 2, 3, 4, 5], [2, 5, 8, 2, 7], size=50)

selected_circle = Circle(fill_alpha=1, fill_color="firebrick", line_color=None)
nonsel_circle=Circle(fill_alpha=0.2,fill_color="blue",line_color="firebrick")

renderer.selection_glyph = selected_circle
renderer.nonselection_glyph = nonsel_circle
```

###### Tool Overlays

Some `Bokeh` tools also have configurable visual attributes. For instance the various region selection tools and box zoom tool all have an overlay whose line and fill properties may be set

```python
x = np.random.random(size=200)
y = np.random.random(size=200)
plot = figure(plot_width=400, plot_height=400, title='Select and Zoom',
              tools="box_select,box_zoom,lasso_select,reset")
plot.circle(x, y, size=5)

select_overlay = plot.select_one(BoxSelectTool).overlay
select_overlay.fill_color = "firebrick"
select_overlay.line_color = None

zoom_overlay = plot.select_one(BoxZoomTool).overla
zoom_overlay.line_color = "olive"
zoom_overlay.line_width = 8

plot.select_one(LassoSelectTool).overlay.line_dash = [10, 10]
```

##### Axes

To set style attributes on Axis objects, use the `xaxis, yaxis`, and `axis` methods on Plot to first obtain a `plot’s` Axis objects. This returns a list of Axis objects. But note that, as convenience, these lists are `splattable`, meaning that you can set attributes directly on this result, and the attributes will be applied to all the axes in the list

###### Labels

The text of an overall label for an axis is controlled by the `axis_label` property. Additionally, there are Text Properties prefixed with `axis_label_` that control the visual appearance of the label. Finally, to change the distance between the axis label and the major tick labels, set the axis_label_standoff property

###### Bounds

Sometimes it is useful to limit the bounds where axes are drawn. This can be accomplished by setting the `bounds` property of an axis object to a 2-tuple of (start, end):

```python
p.xaxis.bounds = (2, 4)
```

###### Tick Locations

`Bokeh` has several “ticker” models that can choose nice locations for ticks. These are configured on the `.ticker` property of an axis. With the `bokeh.plotting` interface, choosing an appropriate ticker type `(categorical, datetime, mercator, linear or log scale)` normally happens automatically. However, there are cases when more explicit control is useful. `FixedTricker`:This ticker model allows users to specify exact tick locations explicitly,

```python
from bokeh.models.tickers import FixedTicker
p.xaxis.ticker = FixedTicker(ticks=[10, 20, 37.4])
```

###### Tick Lines

The visual appearance of the major and minor ticks is controlled by a collection of Line Properties, prefixed with `major_tick_` and `minor_tick_`, respectively. To hide either set of ticks, set the color to None. Additionally, you can control how far in and out of the plotting area the ticks extend, with the properties `major_tick_in/major_tick_out` and `minor_tick_in/minor_tick_out`. These values are in screen units, and negative values are acceptable.

###### Tick Label Formats

The text styling of axis labels is controlled by a `TickFormatter` object configured on the axis’ formatter property. `Bokeh` uses a number of ticker formatters by default in different situations:

`BasicTickFormatter` — Default formatter for linear axes.
`CategoricalTickFormatter` — Default formatter for categorical axes.
`DatetimeTickFormatter` — Default formatter for date time axes.
`LogTickFormatter` — Default formatter for log axes.
These default tick formatters do not expose many configurable properties. To control tick formatting at a finer grained level, use one of the `NumeralTickFormatter` or `PrintfTickFormatter` . The `NumeralTickFormatter` has a format property that can be used to control the text formatting of axis ticks. The `PrintfTickFormatter` has a format property that can be used to control the text formatting of axis ticks using `printf` style format strings.
The `FuncTickFormatter` allows arbitrary tick formatting to be performed by supplying a JavaScript snippet as the code property. For convenience, there are also `from_py_func` and `from_coffeescript` class methods that can convert a python function or `CoffeeScript` snippet into JavaScript automatically. In all cases, the variable tick will contain the unformatted tick value and can be expected to be present in the snippet or function `namespace` at render time. 

```python
p.circle([1,2,3,4,5], [2,5,8,2,7], size=10)

p.xaxis[0].formatter = NumeralTickFormatter(format="0.0%")
p.yaxis[0].formatter = NumeralTickFormatter(format="$0.00")
p.xaxis[0].formatter = PrintfTickFormatter(format="%4.1e")
p.yaxis[0].formatter = PrintfTickFormatter(format="%5.3f mu")
```

###### Tick Label Orientation

The orientation of major tick labels can be controlled with the major_label_orientation property. This property accepts the values **"horizontal" or "vertical"** or a **floating point** number that gives the angle (in radians) to rotate from the horizontal:

##### Grids

Similar to the convenience methods for axes, there are `xgrid, ygrid`, and `grid` methods on Plot that can be used to obtain a `plot’s` Grid objects. These methods also return `splattable` lists, so that you can set an attribute on the list, as if it was a single object, and the attribute is changed for every element of the list

###### Lines

The visual appearance of grid lines is controlled by a collection of Line Properties, prefixed with `grid_`. To hide grid lines, set their line color to None.
The visual appearance of minor grid lines is controlled by a collection of Line Properties, prefixed with `minor_grid_`. By default, minor grid lines are hidden

###### Bands

It is also possible to display filled, shaded bands between adjacent grid lines. The visual appearance of these bands is controlled by a collection of Fill Properties, prefixed with `band_`. To hide grid bands, set their fill color to None .

###### Bounds

Grids also support setting explicit bounds between which they are drawn. They are set in an identical fashion to axes bounds, with a 2-tuple of *(start, end)*

```python
p.xgrid.band_hatch_pattern = "/"
p.xgrid.band_hatch_alpha = 0.6
p.xgrid.band_hatch_color = "lightgrey"

p.grid.bounds = (2, 4)
```

##### Legends

Similar to the convenience methods for axes and grids, there is a `legend` method on Plot that can be used to obtain a `plot’s` Legend objects. This method also returns a `splattable` list, so that you can set an attribute on the list, as if it was a single object, and the attribute is changed for every element of the list

###### Location

The location of the legend labels is controlled by the location property. For legends in the central layout area, values for location can be:

`"top_left", "top_center", "top_right", "center_right", "bottom_right", "bottom_center", "bottom_left", "center_left", "center"`or a (x, y) tuple indicating an absolute location in screen coordinates.
It is also possible to position a legend outside the central area, by using the add_layout method of plots, but doing so requires creating the Legend object directly
The orientation of the legend is controlled by the orientation property. Valid values for this property are:`"vertical", "horizontal"` The default orientation is "vertical".

```python
p.square(x, 3*y, legend="3*sin(x)", fill_color=None, line_color="green")
p.line(x, 3*y, legend="3*sin(x)", line_color="green")

p.legend.location = "bottom_left"

r0 = p.circle(x, y)
r1 = p.line(x, y)
r2 = p.line(x, 2*y, line_dash=[4, 4], line_color="orange", line_width=2)
r3 = p.square(x, 3*y, fill_color=None, line_color="green")
r4 = p.line(x, 3*y, line_color="green")

legend = Legend(items=[("sin(x)"   , [r0, r1]),("2*sin(x)" , [r2]),
    ("3*sin(x)" , [r3, r4]),], location="center")
p.add_layout(legend, 'right')
```

###### Label Text

The visual appearance of the legend labels is controlled by a collection of Text Properties, prefixed with `label_`. The visual appearance of the legend border is controlled by a collection of Line Properties, prefixed with `border_`. To make the border invisible, set the border line color to None. The visual appearance of the legend background is controlled by a collection of Fill Properties, prefixed with `background_`. To make the background transparent, set the background_fill_alpha to 0.

#### Adding Annotations

##### Titles

Title annotations allow descriptive text to be rendered around the edges of a plot. When using `bokeh.plotting` or `bokeh.Charts`, the quickest way to add a basic title is to pass the text as the `title` parameter to Figure or any Chart function. The default `title` is normally on the top of a plot, aligned to the left. But which side of the plot the default title appears on can be controlled by the `title_location` parameter. The default `Title` is accessible through the `Plot.title` property. Visual properties for font, border, background, and others can be set directly on `.title`. In addition to the default title, it is possible to create and add additional Title objects to plots using the `add_layout` method of Plots

##### Legends

It is possible to create Legend annotations easily by specifying a `legend` argument to the glyph methods, when creating a plot. It is also possible to create multiple legend items for the same glyph when if needed by passing a **legend** that is the **column of the column data source.** Other times, it may be useful to explicitly tell `Bokeh` which index into a `ColumnDataSource` should be used when drawing a legend item. In particular, if you want to draw multiple legend items for “multi” glyphs such as `MultiLine` or `Patches.` This is accomplished by **specifying an index** for the legend item 

##### Color Bars

A `ColorBar` can be created using a `ColorMapper` instance, which contains a color palette. Both on- and off-plot color bars are supported; the desired location can be specified when adding the `ColorBar` to the plot.

```python
color_mapper = LogColorMapper(palette="Viridis256", low=1, high=1e7)
color_bar = ColorBar(color_mapper=color_mapper, ticker=LogTicker(),
                     label_standoff=12, border_line_color=None, location=(0,0))

plot.add_layout(color_bar, 'right')
```

##### Arrows

Arrow annotations can be used to connect glyphs and label annotations or to simply highlight plot regions. Arrows are compound annotations, meaning that their start and end attributes are themselves other `ArrowHead` annotations. By default, the Arrow annotation is one-sided with the end set as an `OpenHead-type` arrow head and the start property set to None. Double-sided arrows can be created by setting both the start and end properties as appropriate `ArrowHead` subclass instances. Arrows have standard line properties to set the color and appearance of the arrow shaft. Arrows may also be configured to refer to additional non-default x- or y-ranges with the x_range and y_range properties, in the same way as Twin Axes. Additionally any arrow head objects in start or end have a **size** property to control how big the arrow head is, as well as both **line and fill** properties. The line properties control the outline of the arrow head, and the fill properties control the interior of the arrow head.

```python
from bokeh.models import Arrow, OpenHead, NormalHead, VeeHead
p = figure(plot_width=600, plot_height=600)
p.circle(x=[0, 1, 0.5], y=[0, 0, 0.7], radius=0.1,
         color=["navy", "yellow", "red"], fill_alpha=0.1)

p.add_layout(Arrow(end=OpenHead(line_color="firebrick", line_width=4),
                   x_start=0, y_start=0, x_end=1, y_end=0))
p.add_layout(Arrow(end=NormalHead(fill_color="orange"),
                   x_start=1, y_start=0, x_end=0.5, y_end=0.7))
p.add_layout(Arrow(end=VeeHead(size=35), line_color="red",
                   x_start=0.5, y_start=0.7, x_end=0, y_end=0))
```

##### Bands

A Band will create a dimensionally-linked “stripe”, either located in data or screen coordinates. One common use for the Band annotation is to indicate uncertainty related to a series of measurements.

```python
band = Band(base='x', lower='lower', upper='upper', source=source, level='underlay',
            fill_alpha=1.0, line_width=1, line_color='black')
p.add_layout(band)
```

##### Box Annotations

A `BoxAnnotation` can be linked to either data or screen coordinates in order to emphasize specific plot regions. By default, box annotation dimensions default will extend the annotation to the edge of the plot area.

```python
low_box = BoxAnnotation(top=80, fill_alpha=0.1, fill_color='red')
mid_box = BoxAnnotation(bottom=80, top=180, fill_alpha=0.1, fill_color='green')
high_box = BoxAnnotation(bottom=180, fill_alpha=0.1, fill_color='red')

p.add_layout(low_box)
p.add_layout(mid_box)
p.add_layout(high_box)
```

##### Labels

Labels are text elements that can be used to annotate either glyphs or plot regions. To create a single text label, use the `Label` annotation. This annotation is configured with a text property containing the text to be displayed, as well as x and y properties to set the position. Additionally a render mode `"canvas" or "css" `may be specified. Finally, labels have `text, border_line, and background_fill` properties. These control the visual appearance of the text, as well as the border and background of the bounding box for the text. To create several labels at once, possibly to easily annotate another existing glyph, use the `LabelSet` annotation, which is configured with a data source, with the text and x and y positions are given as column names. `LabelSet` objects can also have **x_offset and y_offset**, which specify a distance in screen space units to offset the label positions from x and y. Finally the render level may be controlled with the level property, to place the label above or underneath other `renderers`

```python
labels = LabelSet(x='weight', y='height', text='names', level='glyph',
              x_offset=5, y_offset=5, source=source, render_mode='canvas')

citation = Label(x=70, y=70, x_units='screen', y_units='screen',
                 text='Collected by Luke C. 2016-04-01', render_mode='css',
                 border_line_color='black', border_line_alpha=1.0,
                 background_fill_color='white', background_fill_alpha=1.0)

p.add_layout(labels)
p.add_layout(citation)
```

##### Slopes

`Slope`annotations are lines which may be sloped and extend to the edge of the plot area.

```python
slope = Slope(gradient=gradient, y_intercept=y_intercept,
              line_color='orange', line_dash='dashed', line_width=3.5)

p.add_layout(slope)
```

##### Spans

`Span` annotations are lines that have a single dimension and extend to the edge of the plot area.

```python
p = figure(x_axis_type="datetime", y_axis_type="datetime")

p.line(daylight_warsaw_2013.Date, daylight_warsaw_2013.Sunset,
       line_dash='solid', line_width=2, legend="Sunset")
p.line(daylight_warsaw_2013.Date, daylight_warsaw_2013.Sunrise,
       line_dash='dotted', line_width=2, legend="Sunrise")

start_date = time.mktime(dt(2013, 3, 31, 2, 0, 0).timetuple())*1000
daylight_savings_start = Span(location=start_date,
                              dimension='height', line_color='green',
                              line_dash='dashed', line_width=3)
p.add_layout(daylight_savings_start)

end_date = dt(2013, 10, 27, 3, 0, 0).timestamp()*1000
daylight_savings_end = Span(location=end_date,
                            dimension='height', line_color='red',
                            line_dash='dashed', line_width=3)
p.add_layout(daylight_savings_end)
```