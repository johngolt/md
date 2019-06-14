#### Plotting with Basic Glyphs

Note that `Bokeh` plots created using the `bokeh.plotting` interface come with a default set of tools, and default visual styles.

##### Scatter Markers

To scatter circle markers on a plot, use the `circle()` method of `Figure`. Similarly, to scatter square markers, use the `square()` method of `Figure`. There are lots of marker types available in `Bokeh`, you can see details and example plots for all of them by clicking on entries in the list below: `asterisk(), circle(), circle_cross(), circle_x(), cross(), dash(), diamond(), diamond_cross(), inverted_triangle(), square(), square_cross(), square_x(), triangle(), x()`. All the markers have the same set of properties: `x, y`, size, and angle. Additionally, `circle()` has a radius property that can be used to specify data-space units.

##### Line Glyphs

generate a single line glyph from one dimensional sequences of x and y points using the `line()` glyph method. For some kinds of data, it may be more appropriate to draw discrete steps between data points, instead of connecting points with linear segments. The `step()` glyph method can be used to accomplish this: Step levels can be drawn before, after, or centered on the x-coordinates, as configured by the mode parameter. Sometimes it is useful to plot multiple lines all at once. This can be accomplished with the `multi_line()` glyph method. `NaN` values can be passed to `line()` and `multi_line()` glyphs. In this case, you end up with single logical line objects, that have multiple disjoint components when rendered.

##### Bars and Rectangles

To draw axis aligned rectangles by specifying the **left, right, top, and bottom** positions, use the `quad()` glyph function. To draw arbitrary rectangles by specifying a **center point, width, height, and angle**, use the `rect()` glyph function. 
When drawing rectangular bars it is often more convenient to have coordinates that are a hybrid of the two systems above. `Bokeh` provides the `hbar()` and `vbar()` glyphs function for this purpose. To draw vertical bars by specifying **a (center) x-coordinate, width, and top and bottom** endpoints, use the `vbar()` glyph function. To draw horizontal bars by specifying **a (center) y-coordinate, height, and left and right** endpoints, use the `hbar()` glyph function:

##### Hex Tiles

`Bokeh` can plot hexagonal tiles, which are often used for showing binned aggregations. The `hex_tile()` method takes a **size** parameter to define the size of the hex grid, and **axial coordinates** to specify which tiles are present.

##### Patch Glyphs

generate a single polygonal patch glyph from one dimensional sequences of x and y points using the `patch()` glyph method. Sometimes it is useful to plot multiple polygonal patches all at once. This can be accomplished with the `patches()` glyph method. Just as with `line()` and `multi_line()`, `NaN` values can be passed to `patch()` and `patches()` glyphs. In this case, you end up with single logical patch objects, that have multiple disjoint components when rendered

##### Polygons with Holes

`p.multi_polygons(xs=[[[[1, 1, 2, 2]]]], ys=[[[[3, 4, 4, 3]]]])`产生一个`polygon`需要一维数组， 在`polygon`中生成`hole`需要二维数组，一组`MultiPolygon`中含有多个`polygon`需要三位数组，产生多个`MultiPolygon`需要四维数组。所以输入的位置参数为一个四维数组。

The `multi_polygons()` glyph uses nesting to accept a variety of information relevant to polygons. Anything that can be rendered as a `patches()` can also be rendered as `multi_polygons()`, but additionally `multi_polygons()` can render holes inside each polygon.Sometimes one conceptual polygon is composed of multiple polygon geometries.The top level of nesting is used to separate each `MultiPolygon` from the others. Each `MultiPolygon` can be thought of as a row in the data source - potentially with a corresponding label or color. 
The `oval()` glyph method accepts the same properties as `rect()`, but renders oval shapes. The `ellipse()` glyph accepts the same properties as `oval()` and `rect()` but renders ellipse shapes, which are different from oval ones. In particular, the same value for width and height will render a circle using the `ellipse()` glyph but not the `oval()` one.

##### Images

You can display images on `Bokeh` plots using the `image()`, `image_rgba()`, and `image_url()` glyph methods.

##### Segments and Rays

Sometimes it is useful to be able to draw many individual line segments at once. `Bokeh` provides the `segment()` and `ray()` glyph methods to render these. The `segment()` function accepts start points $x_0, y_0$ and end points $x_1, y1$ and renders segments between these: The `ray()` function accepts start points $x, y$ with a length and an **angle**. The default angle_units are "rad" but can also be changed to "deg". 

##### Wedges and Arcs

To draw a simple line `arc`, `Bokeh` provides the `arc()` glyph method, which accepts **radius, start_angle, and end_angle** to determine position. Additionally, the **direction** property determines whether to render clockwise or anti-clockwise between the start and end angles. The `wedge()` glyph method accepts the same properties as `arc()`, but renders a filled wedge instead: The `annular_wedge()` glyph method is similar to `arc()`, but draws a filled area. It accepts a **inner_radius and outer_radius** instead of just `radius` Finally, the `annulus()` glyph methods, which accepts **inner_radius and outer_radius**, can be used to draw filled rings. 

##### Specialized Curves

Combining multiple glyphs on a single plot is a matter of calling more than one glyph method on a single Figure. By default, `Bokeh` will attempt to automatically set the data bounds of plots to fit snugly around the data. Sometimes you may need to set a plot’s range explicitly. This can be accomplished by setting the **x_range or y_range** properties using a `Range1d` object that gives the start and end points of the range you want. As a convenience, the `figure()` function can also accept tuples of (start, end) as values for the **x_range or y_range** parameters. Categorical axes are created by specifying a `FactorRange` for one of the plot ranges (or a lists of factors to be converted to one).  When dealing with time series data, or any data that involves dates or times, it is desirable to have an axis that can display labels that are appropriate to different date and time scales. We have seen how to use the `figure()` function to create plots using the `bokeh.plotting` interface. This function accepts **x_axis_type and y_axis_type** as arguments. To specify a `datetime` axis, pass `"datetime"` for the value of either of these parameters. When dealing with data that grows exponentially or is of many orders of magnitude, it is often necessary to have one axis on a log scale. Another scenario involves plotting data that has a power law relationship, when it is desirable to use log scales on both axes. As we saw above, the `figure()` function accepts **x_axis_type and y_axis_type** as arguments. To specify a log axis, pass` "log"` for the value of either of these parameters. It is possible to add multiple axes representing different ranges to a single plot. To do this, configure the plot with `“extra”` named ranges in the **extra_x_range and extra_y_range** properties. Then these named ranges can be referred to when adding new glyph methods, and also to add new axes objects using the add_layout method on Plot. 

#### Providing Data for Plots and Tables

##### Providing data directly

In `Bokeh`, it is possible to pass lists of values directly into plotting functions.When you pass in data like this, `Bokeh` works behind the scenes to make a `ColumnDataSource` for you. 

##### `ColumnDataSource​`

The `ColumnDataSource` is the core of most `Bokeh` plots, providing the data that is visualized by the glyphs of the plot. With the `ColumnDataSource`, it is easy to share data between multiple plots and widgets. When the same `ColumnDataSource` is used to drive multiple `renderers`, selections of the data source are also shared. Thus it is possible to use a select tool to choose data points from one plot and have them automatically highlighted in a second plot.

At the most basic level, a `ColumnDataSource` is simply a mapping between column names and lists of data. The `ColumnDataSource` takes a data parameter which is a `dict`, with string column names as keys and lists (or arrays) of data values as values. If one positional argument is passed in to the `ColumnDataSource` initializer, it will be taken as data. Once the `ColumnDataSource` has been created, it can be passed into the **source** parameter of plotting methods which allows you to pass a column’s name as a stand in for the data values.

###### Pandas

If a `DataFrame` is used, the `CDS` will have columns corresponding to the columns of the `DataFrame`. The index of the `DataFrame` will be reset, so if the `DataFrame` has a named index column, then `CDS` will also have a column with this name. However, if the index name is `None`, then the `CDS` will be assigned a generic name. It will be **index** if it is available, and `level_0` otherwise.

###### Pandas `MultiIndex`

All `MultiIndex` columns and `indices` will be flattened before forming the `ColumnsDataSource`. For the index, **an index of tuples** will be created, and the names of the `MultiIndex` joined with an **underscore**. The column names will also be joined with an underscore. 

###### Pandas `GroupBy`

If a `GroupBy` object is used, the `CDS` will have columns corresponding to the result of calling `group.describe()`. The describe method generates columns for statistical measures for all the non-grouped original columns. The resulting `DataFrame` has `MultiIndex` columns with the original column name and the computed measure, so it will be flattened using the aforementioned scheme. 

###### Streaming

`ColumnDataSource` streaming is an efficient way to append new data to a `CDS`. By using the stream method, `Bokeh` only sends new data to the browser instead of the entire dataset. The stream method takes a `new_data` parameter containing a `dict` mapping column names to sequences of data to be appended to the respective columns. It additionally takes an optional argument `rollover`, which is the maximum length of data to keep. The default rollover value of None allows data to grow unbounded.

###### Patching

`ColumnDataSource` patching is an efficient way to update slices of a data source. By using the patch method, `Bokeh` only needs to send new data to the browser instead of the entire dataset. The patch method should be passed a `dict` mapping column names to list of tuples that represent a patch change to apply.

##### Transforming Data

We have seen above how data can be added to a `ColumnDataSource` to drive `Bokeh` plots. This can include raw data or data that we explicitly transform ourselves, for example a column of colors created to control how the Markers in a scatter plot should be shaded. It is also possible to specify transforms that only occur in the browser. This can be useful to reduce both code as well as the amount of data that has to be sent into the browser.

###### Colors

To perform linear color mapping in the browser, the `linear_cmap()` function may be used. It accepts the name of a `ColumnDataSource` column to `colormap`, a palette (which can be a built-in palette name, or an actual list of colors), and min/max values for the color mapping range. The result can be passed to a **color** property on glyphs:

###### Markers

It is also possible to map categorical data to marker types. the use of `factor_mark()` to display different markers or different categories in the input data. the use of `factor_cmap()` to `colormap` those same categories

##### Filtering Data

It’s often desirable to focus in on a portion of data that has been `subsampled` or filtered from a larger dataset. `Bokeh` allows you to specify a view of a data source that represents a subset of data. By having a view of the data source, the underlying data does not need to be changed and can be shared across plots. The view consists of one or more filters that select the rows of the data source that should be bound to a specific glyph. To plot with a subset of data, you can create a `CDSView` and pass it in as a **view** argument to the renderer-adding methods on the Figure. The `CDSView` has two properties, **source and filters**. source is the `ColumnDataSource` that the view is associated with. filters is a list of Filter objects.

###### `IndexFilter`

The `IndexFilter` is the simplest filter type. It has an `indices` property which is a list of integers that are the `indices` of the data you want to be included in the plot.

###### `BooleanFilter`

A `BooleanFilter` selects rows from a data source through a list of True or False values in its booleans property.

###### `GroupFilter`

The `GroupFilter` allows you to select rows from a dataset that have a specific value for a categorical variable. The `GroupFilter` has two properties, **column_name**, the name of column in the `ColumnDataSource`, and **group**, the value of the column to select for.

##### Linked Selection

With the ability to specify a subset of data to be used for each glyph renderer, it is easy to share data between plots even when the plots use different subsets of data. By using the same `ColumnDataSource`, selections and hovered inspections of that data source are automatically shared. Selections in either plot are automatically reflected in the other. And hovering on a point in one plot will highlight the corresponding point in the other plot if it exists.

#### Laying out Plots and Widgets



#### Handling Categorical Data

##### Bars

Since `Bokeh` displays bars in the order the factors are given for the range, “sorting” bars in a bar plot is identical to sorting the factors for the range.
Often times we may want to have bars that are shaded some color. This can be accomplished in different ways. One way is to supply all the colors up front. This can be done by putting all the data, including the colors for each bar, in a `ColumnDataSource`. Then the name of the column containing the colors is passed to figure as the color arguments. 

###### Grouped

When creating bar charts, it is often desirable to visually display the data according to sub-groups. There are two basic methods that can be used, depending on your use case: using nested categorical coordinates, or applying dodges. If the coordinates of a plot range and data have two or three levels, then `Bokeh` will automatically group the factors on the axis, including a hierarchical tick labeling with separators between the groups. In the case of bar charts, this results in bars grouped together by the top-level factors. This is probably the most common way to achieve grouped bars, especially if you are starting from “tidy” data.
Another method for achieving grouped bars is to explicitly specify a visual displacement for the bars. Such a visual offset is also referred to as a dodge. In this scenario, our data is not “tidy”. Instead a single table with rows indexed by factors (fruit, year), we have separate series for each year. We can plot all the year series using separate calls to `vbar` but since every bar in each group has the same fruit factor, the bars would overlap visually. We can prevent this overlap and distinguish the bars visually by using the `dodge()` function to provide an offset for each different call to `vbar`

Another common operation or bar charts is to stack bars on top of one another. `Bokeh` makes this easy to do with the specialized `hbar_stack()` and `vbar_stack()` functions. 
Note that behind the scenes, these functions work by stacking up the successive columns in separate calls to `vbar` or `hbar`. This kind of operation is akin the to dodge example above. Sometimes we may want to stack bars that have both positive and negative extents. it is possible to create such a stacked bar chart that is split by positive and negative values. For stacked bar plots, `Bokeh` provides some special hover variables that are useful for common cases. When stacking bars, `Bokeh` automatically sets the name property for each layer in the stack to be the value of the stack column for that layer. This name value is accessible to hover tools via the `$name` special variable. Additionally, the hover variable `@$name` can be used to look up values from the stack column for each layer. For instance, if a user hovers over a stack glyph with the name "US East", then `@$name` is equivalent to `@{US East}`. Note that it is also possible to override the value of name by passing it manually to `vbar_stack` and `hbar_stack`. In this case, `$@name` will look up the column names provided by the user. It may also sometimes be desirable to have a different hover tool for each layer in the stack. For such cases, the `hbar_stack` and `vbar_stack` functions return a list of all the `renderers` created. These can be used to customize different hover tools for each layer

##### Mixed Factors



#### Configuring Plot Tools

`Bokeh` comes with a number of interactive tools that can be used to report information, to change plot parameters such as zoom level or range extents, or to add, edit, or delete glyphs. Tools can be grouped into four basic categories: Gestures--These are tools that respond to single gestures, such as a pan movement. The types of gesture tools are: Pan/Drag Tools, Click/Tap Tools, Scroll/Pinch Tools. For each type of gesture, one tool can be active at any given time, and the active tool is indicated on the `toolbar` by a highlight next to to the tool icon;  Actions: These are immediate or modal operations that are only activated when their button in the `toolbar` is pressed;  Inspectors--These are passive tools that report information or annotate plots in some way;  Edit Tools--These are sophisticated multi-gesture tools that can add, delete, or modify glyphs on a plot. Since they may respond to several gestures at once, an edit tool will potentially deactivate multiple single-gesture tools at once when it is activated.

By default, `Bokeh` plots come with a `toolbar` above the plot. The `toolbar` location can be specified by passing the` toolbar_location` parameter to the `figure()` function. Valid values are: `"above", "below", "left", "right"`. If you would like to hide the `toolbar` entirely, pass None. Note that the `toolbar` position clashes with the default axes, in this case setting the `toolbar_sticky` option to False will move the `toolbar` to outside of the region where the axis is drawn. At the lowest `bokeh.models` level, tools are added to a Plot by passing instances of `Tool` objects to the `add_tools` method. This explicit way of adding tools works with any `Bokeh` Plot or Plot subclass, such as Figure. Tools can be specified by passing the tools parameter to the `figure()` function. The tools parameter accepts a list of tool objects.  Tools can also be supplied conveniently with a comma-separate string containing tool shortcut names. Finally, it is also always possible to add new tools to a plot by passing a tool object to the `add_tools` method of a plot. This can also be done in conjunction with the tools keyword

Bokeh comes with a number of interactive tools that can be used to report information, to change plot parameters such as zoom level or range extents, or to add, edit, or delete glyphs. Tools can be grouped into four basic categories:
Gestures
These are tools that respond to single gestures, such as a pan movement. The types of gesture tools are:

Pan/Drag Tools
Click/Tap Tools
Scroll/Pinch Tools
For each type of gesture, one tool can be active at any given time, and the active tool is indicated on the toolbar by a highlight next to to the tool icon.

Actions
These are immediate or modal operations that are only activated when their button in the toolbar is pressed, such as the ResetTool.
Inspectors
These are passive tools that report information or annotate plots in some way, such as the HoverTool or CrosshairTool.
Edit Tools
These are sophisticated multi-gesture tools that can add, delete, or modify glyphs on a plot. Since they may respond to several gestures at once, an edit tool will potentially deactivate multiple single-gesture tools at once when it is activated.

By default, Bokeh plots come with a toolbar above the plot. In this section you will learn how to specify a different location for the toolbar, or to remove it entirely.

The toolbar location can be specified by passing the toolbar_location parameter to the figure() function. Valid values are:

"above"
"below"
"left"
"right"
If you would like to hide the toolbar entirely, pass None.
Note that the toolbar position clashes with the default axes, in this case setting the toolbar_sticky option to False will move the toolbar to outside of the region where the axis is drawn.
At the lowest bokeh.models level, tools are added to a Plot by passing instances of Tool objects to the add_tools method:
This explicit way of adding tools works with any Bokeh Plot or Plot subclass, such as Figure.

Tools can be specified by passing the tools parameter to the figure() function. The tools parameter accepts a list of tool objects,
Tools can also be supplied conveniently with a comma-separate string containing tool shortcut names:
Finally, it is also always possible to add new tools to a plot by passing a tool object to the add_tools method of a plot. This can also be done in conjunction with the tools keyword 

##### Gestures

######  Pan/Drag Tools

| 函数              | name         | 作用                                                         |
| ----------------- | ------------ | ------------------------------------------------------------ |
| `BoxSelectTool`   | box_select   | The box selection tool allows the user to define a rectangular selection region by left-dragging a mouse, or dragging a finger across the plot area. |
| `BoxZoomTool`     | box_zoom     | The box zoom tool allows the user to define a rectangular region to zoom the plot bounds too, by left-dragging a mouse, or dragging a finger across the plot area. |
| `LassoSelectTool` | lasso_select | The lasso selection tool allows the user to define an arbitrary region for selection by left-dragging a mouse, or dragging a finger across the plot area. |
| `PanTool`         | pan          | The pan tool allows the user to pan the plot by left-dragging a mouse or dragging a finger across the plot region. |

###### Click/Tap Tools

| 函数             | name        | 作用                                                         |
| ---------------- | ----------- | ------------------------------------------------------------ |
| `PolySelectTool` | poly_select | The polygon selection tool allows the user to define an arbitrary polygonal region for selection by left-clicking a mouse, or tapping a finger at different locations. Complete the selection by making a double left-click or tapping. To make a multiple selection, press the SHIFT key. To clear the selection, press the `ESC` key. |
| `TapTool`        | tap         | The tap selection tool allows the user to select at single points by clicking a left mouse button, or tapping with a finger. |

###### Scroll/Pinch Tools

| 函数            | name         | 作用                                                         |
| --------------- | ------------ | ------------------------------------------------------------ |
| `WheelZoomTool` | wheel_zoom   | The wheel zoom tool will zoom the plot in and out, centered on the current mouse location. It will respect any min and max values and ranges preventing zooming in and out beyond these. |
| `WheelPanTool`  | `xwheel_pan` | The wheel pan tool will translate the plot window along the specified dimension without changing the window’s aspect ratio. The tool will respect any min and max values and ranges preventing panning beyond these values. |

##### Actions

| 函数          | name     | 作用                                                         |
| ------------- | -------- | ------------------------------------------------------------ |
| `UndoTool`    | undo     | The undo tool allows to restore previous state of the plot.  |
| `RedoTool`    | redo     | The redo tool reverses the last action performed by undo tool. |
| `ResetTool`   | reset    | The reset tool will restore the plot ranges to their original values. |
| `SaveTool`    | save     | The save tool pops up a modal dialog that allows the user to save a `PNG` image of the plot. |
| `ZoomInTool`  | zoom_in  | The zoom-in tool will increase the zoom of the plot. It will respect any min and max values and ranges preventing zooming in and out beyond these. |
| `ZoomOutTool` | zoom_out | The zoom-out tool will decrease the zoom level of the plot. It will respect any min and max values and ranges preventing zooming in and out beyond these. |

##### Inspectors

| 函数            | name        | 作用                                                         |
| --------------- | ----------- | ------------------------------------------------------------ |
| `CrosshairTool` | `crosshair` | Th `crosshair` tool draws a `crosshair` annotation over the plot, centered on the current mouse position. |
| `HoverTool`     | hover       | The hover tool is a passive inspector tool.                  |

By default, the hover tool will generate a “tabular” tooltip where each row contains a label, and it's associated value. The labels and values are supplied as a list of (label, value) tuples. Field names that begin with `$` are “special fields”. These often correspond to values that are intrinsic to the plot, such as the coordinates of the mouse in data or screen space. These special fields are listed here:

`$index`:	index of selected point in the data source
`$name`:	value of the name property of the hovered glyph renderer
`$x`:	x-coordinate under the cursor in data space
`$y`:	y-coordinate under the cursor in data space
​`$sx`:	x-coordinate under the cursor in screen (canvas) space
`$sy`:	y-coordinate under the cursor in screen (canvas) space
`$name`:	The name property of the glyph that is hovered over
`$color`:	colors from a data source
Field names that begin with `@` are associated with columns in a `ColumnDataSource`. For instance the field name "@price" will display values from the "price" column whenever a hover is triggered. If the hover is for the 17th glyph, then the hover `tooltip` will correspondingly display the 17th price value. Note that if a column name contains spaces, the it must be supplied by surrounding it in curly braces. Sometimes it is desirable to allow the name of the column be specified indirectly. The field name `@$name` is distinguished in that it will look up the name field on the hovered glyph renderer, and use that value as the column name. 
The hover tool displays informational `tooltips` associated with individual glyphs. These `tooltips` can be configured to activate in in different ways with a **mode** property: `"mouse"`: only when the mouse is directly over a glyph; `"vline"`:	whenever the a vertical line from the mouse position intersects a glyph; `"hline"`: whenever the a horizontal line from the mouse position intersects a glyph

By default, values for fields are displayed in a basic numeric format. However it is possible to control the formatting of values more precisely. Fields can be modified by appending a format specified to the end in curly braces. there are other formatting schemes that can be specified for interpreting format strings: `"numeral"`:	Provides a wide variety of formats for numbers, currency, bytes, times, and percentages. 
`"datetime"`:	Provides formats for date and time values; `"printf"`:	Provides formats similar to C-style `“printf”` type specifiers.

#### Styling Visual Attributes

##### Color

Palettes are sequences (lists or tuples) of `RGB(A)` hex strings that define a `colormap` and be can set as the color attribute of many plot objects from `bokeh.plotting`. `Bokeh` offers many of the standard Brewer palettes, which can be imported from the `bokeh.palettes` module. Color Mappers allow you to encode some data sequence into a palette of colors based on the value in that sequence. The mappers are then set as the color attribute on marker objects. `Bokeh` includes several types of mappers to encode colors: `bokeh.transform.factor_cmap`: Maps colors to specific categorical elements; `bokeh.transform.linear_cmap`: Maps a range of numeric values across the available colors from high to low. `bokeh.transform.log_cmap`: Similar to `linear_cma`p but uses a natural log scale to map the colors. These mapper functions return a `DataSpec` property that can be passed to the color attribute of the glyph. The returned `dataspec` includes a `bokeh.transform` which can be accessed to use the mapper in another context such as to create a `ColorBar` as in the example below:
Colors properties are used in many places in `Bokeh`, to specify the colors to use for lines, fills or text. Color values can be provided in any of the following ways: any of the 147 named `CSS` colors; an `RGB(A)` hex value, e.g., `'#FF0000'`; a 3-tuple of integers (r,g,b) between 0 and 255; a 4-tuple of `(r,g,b,a)` where `r, g, b` are integers between 0 and 255 and `a` is a floating point value between 0 and 1

There are several `ArrowHead` subtypes that can be applied to Arrow annotations. Setting the start or end property to `None` will cause no arrow head to be applied at the specified arrow end. Double-sided arrows can be created by setting both start and end styles. Setting visible to false on an arrow will also make the corresponding arrow head invisible.
`Screen units` use raw numbers of pixels to specify height or width, while data-space units are relative to the data and the axes of the plot. 

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

##### Plots

`Bokeh` plots comprise graphs of objects that represent all the different parts of the plot: grids, axes, glyphs, etc. In order to style `Bokeh` plots, it is necessary to first find the right object, then set its various attributes. Some objects have convenience methods to help find the objects of interest . But there is also a `select()` method on Plot that can be used to query for `Bokeh` plot objects more generally.

Plot objects themselves have many visual characteristics that can be styled: the dimensions of the plot, backgrounds, borders, outlines, etc. 

###### Dimensions

The dimensions (width and height) of a Plot are controlled by `plot_width` and `plot_height` attributes. These values are in screen units, and they control the size of the entire canvas area, including any axes or titles. If you are using the `bokeh.plotting` interface, then these values can be passed to `figure()` as a convenience

###### Title

The styling of the plot title is controlled by the properties of `Title` annotation, which is available as the `.title` property on the Plot. Most of the standard `Text` Properties are available, with the exception of `text_align` and `text_baseline` which do not apply. For positioning the title relative to the entire plot, use the properties **align and offset**.

###### Background

The background fill style is controlled by the `background_fill_color` and `background_fill_alpha` properties of the Plot object

###### Border

The border fill style is controlled by the `border_fill_color` and `border_fill_alpha` properties of the Plot object. You can also set the minimum border on each side with the properties: `min_border_left, min_border_right, min_border_top, min_border_bottom`. Additionally, setting `min_border` will apply a minimum border setting to all sides as a convenience.

###### Outline

The styling of the outline of the plotting area is controlled by a set of Line Properties on the Plot, that are prefixed with `outline_`.

###### Glyphs

To style the fill, line, or text properties of a glyph, it is first necessary to obtain a specific `GlyphRenderer`. When using the `bokeh.plotting` interface, the glyph functions return the renderer. Then, the glyph itself is obtained from the `.glyph` attribute of a `GlyphRenderer`. This is the object to set fill, line, or text property values for.

###### Selected and Unselected Glyphs

The styling of selected and non-selected glyphs can be customized by setting the `selection_glyph` and/or `nonselection_glyph` attributes of the `GlyphRenderer` either manually or by passing them to `add_glyph()`. Click or tap circles on the plot to see the effect on the selected and non-selected glyphs. To clear the selection and restore the original state, click anywhere in the plot outside of a circle.
If you just need to set the **color or alpha** parameters of the selected or non-selected glyphs, this can be accomplished even more simply by providing color and alpha arguments to the glyph function, prefixed by `"selection_" or "nonselection_"`. 

##### Axes

To set style attributes on Axis objects, use the `xaxis, yaxis`, and `axis` methods on Plot to first obtain a plot’s Axis objects. This returns a list of Axis objects. But note that, as convenience, these lists are `splattable`, meaning that you can set attributes directly on this result, and the attributes will be applied to all the axes in the list

###### Labels

The text of an overall label for an axis is controlled by the `axis_label` property. Additionally, there are Text Properties prefixed with `axis_label_` that control the visual appearance of the label. Finally, to change the distance between the axis label and the major tick labels, set the axis_label_standoff property

###### Bounds

Sometimes it is useful to limit the bounds where axes are drawn. This can be accomplished by setting the `bounds` property of an axis object to a 2-tuple of (start, end):

###### Tick Locations

`Bokeh` has several “ticker” models that can choose nice locations for ticks. These are configured on the `.ticker` property of an axis. With the `bokeh.plotting` interface, choosing an appropriate ticker type `(categorical, datetime, mercator, linear or log scale)` normally happens automatically. However, there are cases when more explicit control is useful. `FixedTricker`:This ticker model allows users to specify exact tick locations explicitly,

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

###### Tick Label Orientation

The orientation of major tick labels can be controlled with the major_label_orientation property. This property accepts the values **"horizontal" or "vertical"** or a **floating point** number that gives the angle (in radians) to rotate from the horizontal:

##### Grids

Similar to the convenience methods for axes, there are `xgrid, ygrid`, and `grid` methods on Plot that can be used to obtain a plot’s Grid objects. These methods also return `splattable` lists, so that you can set an attribute on the list, as if it was a single object, and the attribute is changed for every element of the list

###### Lines

The visual appearance of grid lines is controlled by a collection of Line Properties, prefixed with `grid_`. To hide grid lines, set their line color to None.
The visual appearance of minor grid lines is controlled by a collection of Line Properties, prefixed with `minor_grid_`. By default, minor grid lines are hidden

###### Bands

It is also possible to display filled, shaded bands between adjacent grid lines. The visual appearance of these bands is controlled by a collection of Fill Properties, prefixed with `band_`. To hide grid bands, set their fill color to None .

###### Bounds

Grids also support setting explicit bounds between which they are drawn. They are set in an identical fashion to axes bounds, with a 2-tuple of *(start, end)*

##### Legends

Similar to the convenience methods for axes and grids, there is a `legend` method on Plot that can be used to obtain a plot’s Legend objects. This method also returns a `splattable` list, so that you can set an attribute on the list, as if it was a single object, and the attribute is changed for every element of the list

###### Location

The location of the legend labels is controlled by the location property. For legends in the central layout area, values for location can be:

`"top_left", "top_center", "top_right", "center_right", "bottom_right", "bottom_center", "bottom_left", "center_left", "center"`or a (x, y) tuple indicating an absolute location in screen coordinates.
It is also possible to position a legend outside the central area, by using the add_layout method of plots, but doing so requires creating the Legend object directly
The orientation of the legend is controlled by the orientation property. Valid values for this property are:`"vertical", "horizontal"` The default orientation is "vertical".

###### Label Text

The visual appearance of the legend labels is controlled by a collection of Text Properties, prefixed with `label_`. The visual appearance of the legend border is controlled by a collection of Line Properties, prefixed with `border_`. To make the border invisible, set the border line color to None. The visual appearance of the legend background is controlled by a collection of Fill Properties, prefixed with `background_`. To make the background transparent, set the background_fill_alpha to 0.

#### Adding Annotations

##### Titles

Title annotations allow descriptive text to be rendered around the edges of a plot. When using `bokeh.plotting` or `bokeh.Charts`, the quickest way to add a basic title is to pass the text as the `title` parameter to Figure or any Chart function. The default `title` is normally on the top of a plot, aligned to the left. But which side of the plot the default title appears on can be controlled by the `title_location` parameter. The default `Title` is accessible through the `Plot.title` property. Visual properties for font, border, background, and others can be set directly on `.title`. In addition to the default title, it is possible to create and add additional Title objects to plots using the `add_layout` method of Plots

##### Legends

It is possible to create Legend annotations easily by specifying a `legend` argument to the glyph methods, when creating a plot. It is also possible to create multiple legend items for the same glyph when if needed by passing a **legend** that is the **column of the column data source.** Other times, it may be useful to explicitly tell `Bokeh` which index into a `ColumnDataSource` should be used when drawing a legend item. In particular, if you want to draw multiple legend items for “multi” glyphs such as `MultiLine` or `Patches.` This is accomplished by **specifying an index** for the legend item 

##### Color Bars

A `ColorBar` can be created using a `ColorMapper` instance, which contains a color palette. Both on- and off-plot color bars are supported; the desired location can be specified when adding the `ColorBar` to the plot.

##### Arrows

Arrow annotations can be used to connect glyphs and label annotations or to simply highlight plot regions. Arrows are compound annotations, meaning that their start and end attributes are themselves other `ArrowHead` annotations. By default, the Arrow annotation is one-sided with the end set as an `OpenHead-type` arrow head and the start property set to None. Double-sided arrows can be created by setting both the start and end properties as appropriate `ArrowHead` subclass instances. Arrows have standard line properties to set the color and appearance of the arrow shaft. Arrows may also be configured to refer to additional non-default x- or y-ranges with the x_range and y_range properties, in the same way as Twin Axes. Additionally any arrow head objects in start or end have a **size** property to control how big the arrow head is, as well as both **line and fill** properties. The line properties control the outline of the arrow head, and the fill properties control the interior of the arrow head.

##### Bands

A Band will create a dimensionally-linked “stripe”, either located in data or screen coordinates. One common use for the Band annotation is to indicate uncertainty related to a series of measurements.

##### Box Annotations

A `BoxAnnotation` can be linked to either data or screen coordinates in order to emphasize specific plot regions. By default, box annotation dimensions default will extend the annotation to the edge of the plot area.

##### Labels

Labels are text elements that can be used to annotate either glyphs or plot regions. To create a single text label, use the `Label` annotation. This annotation is configured with a text property containing the text to be displayed, as well as x and y properties to set the position. Additionally a render mode `"canvas" or "css" `may be specified. Finally, labels have `text, border_line, and background_fill` properties. These control the visual appearance of the text, as well as the border and background of the bounding box for the text. To create several labels at once, possibly to easily annotate another existing glyph, use the `LabelSet` annotation, which is configured with a data source, with the text and x and y positions are given as column names. `LabelSet` objects can also have **x_offset and y_offset**, which specify a distance in screen space units to offset the label positions from x and y. Finally the render level may be controlled with the level property, to place the label above or underneath other `renderers`

##### Slopes

`Slope`annotations are lines which may be sloped and extend to the edge of the plot area.

##### Spans

`Span` annotations are lines that have a single dimension and extend to the edge of the plot area.

