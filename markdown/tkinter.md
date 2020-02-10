#### Geometry Management

Geometry management in `Tk` relies on the concept of *master* and *slave* widgets.  A master is a widget, typically a `toplevel` window or a frame, which will contain other widgets, which are called slaves.  You can think of a geometry manager as taking control of the master widget, and deciding what will be displayed within. The geometry manager will ask each slave widget for its natural size, or how large it would ideally like to be displayed. It then takes that information and combines it with any parameters provided by the program when it asks the geometry manager to manage that particular slave widget.  In our example, we passed grid a `"column"` and `"row"` number for each widget, which indicated the relative position of the widget with respect to others, and also a `"sticky"` parameter to suggest how the widget should be aligned or possibly stretched.  We also used `"columnconfigure"` and `"rowconfigure"` to indicate the columns and rows we'd like to expand if there is extra space available in the window.  Of course, all these parameters are specific to grid; other geometry managers would use different ones.

  The geometry manager takes all the information about the slaves, as well as the information about how large the master is, and uses its internal algorithms to determine the area each slave will be allocated.  The slave is then responsible for drawing within that particular rectangle.  And of course, any time the size of the master changes, the natural size of a slave changes. because we've changed the text in a label, or any of the geometry manager parameters change we repeat the whole thing.

  This all works recursively as well.  In our example, we had a content frame inside the toplevel window, and then a number of other controls in the content frame.  We, therefore, had a geometry manager working on two different masters.  At the outer level, the toplevel window was the master, and the content frame was the slave.  At the inner level, the content frame was the master, with each of the other widgets being slaves.  So the same widget can be both a master and a slave.  This hierarchy can, of course, also be nested much more deeply.

#### Event Handling

##### Command Callbacks

Often though you want your program to handle particular events, for example doing something when a button is  pushed.  For those events that are pretty much essential to customize (what good is a button without something happening when you press it?), the widget will provide a *callback* as a widget configuration option. We saw this in the example with the `"command"` option of the button.  

  Callbacks in Tk tend to be simpler than in toolkits used with compiled languages. Instead, the callback is just a normal bit of code that the interpreter evaluates.  While it can be as complex as you want to make it, most times though you'll just want your callback to call some other procedure.

##### Event Bindings

For events that don't have a command callback associated with them, you can use `Tk's "bind"` to capture any event, and then (like with callbacks) execute an arbitrary piece of code. 

  #### 核心组件

| 名称        | 作用                                          |
| ----------- | --------------------------------------------- |
| Button      | 按钮                                          |
| Canvas      | 绘图形组建，可以在其中绘制图形                |
| Checkbutton | 复选框                                        |
| Entry       | 文本框单行                                    |
| Text        | 文本框多行                                    |
| Frame       | 框架，将几个组件组成一组                      |
| Label       | 标签，可以显示文字或图片                      |
| Listbox     | 列表框                                        |
| Menu        | 菜单                                          |
| Menubutton  | 跟Menu一样                                    |
| Message     | 与Label组件类似，但可以根据自身大小将文本换行 |
| Radiobutton | 单选框                                        |
| Scale       | 滑块；允许通过滑块来设置数字值                |
| Scrollbar   | 滚动条；配合其他使用                          |
| Toplevel    | 用来创建子窗口窗口组件                        |

##### `pack`组件设置

| 属性   | 作用                                                         |
| ------ | ------------------------------------------------------------ |
| after  | 将组件置于其他组件之后                                       |
| before | 将组件置于其他组件之前                                       |
| anchor | 组件的对齐方式，顶对齐'n',底对齐's',左'w',右'e'              |
| side   | 组件在主窗口的位置，可以为'top','bottom','left','right'（使用时`tkinter.TOP`,`tkinter.E`） |
| fill   | 填充方式 (Y,垂直，X，水平）                                  |
| expand | 1可扩展，0不可扩展                                           |

##### `grid`组件

| 参数         | 作用                 |
| ------------ | -------------------- |
| column       | 组件所在的列起始位置 |
| `columnspam` | 组件的列宽           |
| row          | 组件所在的行起始位置 |
| `rowspam`    | 组件的行宽           |

##### `place`组件

| 参数                  | 作用                                        |
| --------------------- | ------------------------------------------- |
| anchor                | 组件对齐方式                                |
| x, y                  | 组件左上角的x, y坐标                        |
| `relx, rely`          | 组件相对于窗口的x, y坐标，应为0-1之间的小数 |
| `width, height`       | 组件的宽度，高度                            |
| `relwidth, relheight` | 组件相对于窗口的宽度、高度，0-1             |

##### `tkinter.Button`

| 参数                             | 作用                                  |
| -------------------------------- | ------------------------------------- |
| anchor                           | 指定按钮上文本的位置；                |
| `background(bg), foreground(fg)` | 指定按钮的背景色、前景色              |
| `bitmap`                         | 指定按钮上显示的位图                  |
| `borderwidth(bd)`                | 指定按钮边框的宽度                    |
| `command`                        | 指定按钮消息的回调函数                |
| `cursor`                         | 指定鼠标移动到按钮上的指针样式        |
| `font`                           | 指定按钮上文本的字体                  |
| `height, width`                  | 指定按钮的高度、宽度                  |
| `image`                          | 指定按钮上显示的图片                  |
| `state`                          | 指定按钮的状态（disabled）            |
| `text`                           | 指定按钮上显示的文本                  |
| `padx`                           | 设置文本与按钮边框x的距离，还有`pady` |
| `activeforeground`               | 按下时前景色                          |
| `textvariable`                   | 可变文本，与`StringVar`等配合着用     |

##### `tkinter.Entry, tkinter.Text`

| 参数               | 作用                                        |
| ------------------ | ------------------------------------------- |
| `bg,fg`            | 文本框背景色、前景色；                      |
| `selectbackground` | 选定文本背景色                              |
| `selectforeground` | 选定文本前景色                              |
| `borderwidth`      | 文本框边框宽度                              |
| `font`             | 字体                                        |
| `show`             | 文本框显示的字符，若为*，表示文本框为密码框 |
| `state`            | 状态                                        |
| `width`            | 文本框宽度                                  |

##### `tkinter.Label`

| 参数                                                       | 作用               |
| ---------------------------------------------------------- | ------------------ |
| `anchor, bg, fg, bd, width, bitmap, font, image, text,...` |                    |
| `justify`                                                  | 多行文本的对齐方式 |

##### `thinter.Radiobutton, Checkbutton`

| 参数                                                         | 作用                                          |
| ------------------------------------------------------------ | --------------------------------------------- |
| `anchor, bg, fg, bd, width, bitmap, image, font, justify, text,` |                                               |
| `value`                                                      | 指定组件被选中中关联变量的值                  |
| `variable`                                                   | 指定组件所关联的变量                          |
| `indicatoron`                                                | 特殊控制参数，当为0时，组件会被绘制成按钮形式 |

##### Canvas

| 参数                               | 作用 |
| ---------------------------------- | ---- |
| `bg, fg, bd, width, bitmap, image` |      |

绘图的方法主要以下几种：
    create_arc          圆弧;
    create_bitmap  　　  绘制位图，支持XBM;
    create_image    　　 绘制图片，支持GIF(x,y,image,anchor);
    create_line         绘制支线；
    create_oval;        绘制椭圆；
    create_polygon   　　绘制多边形(坐标依次罗列，不用加括号，还有参数，fill,outline)；
    create_rectangle　　 绘制矩形((a,b,c,d),值为左上角和右下角的坐标)；
    create_text         绘制文字(字体参数font,)；
    create_window    　　绘制窗口；
    delete            　 删除绘制的图形；
    itemconfig          修改图形属性，第一个参数为图形的ID，后边为想修改的参数；
    move          　　   移动图像（1，4，0），1为图像对象，4为横移4像素，0为纵移像素，然后用root.update()刷新即可看到图像的移动，为了使多次移动变得可视，最好加上time.sleep()函数；
    只要用create_方法画了一个图形，就会自动返回一个ID,创建一个图形时将它赋值给一个变量，需要ID时就可以使用这个变量名。
    coords(ID)          返回对象的位置的两个坐标（4个数字元组）；

对于按钮组件、菜单组件等可以在创建组件时通过command参数指定其事件处理函数。方法为bind;或者用bind_class方法进行类绑定，bind_all方法将所有组件事件绑定到事件响应函数上。

##### Menu

| 参数                                 | 作用                                 |
| ------------------------------------ | ------------------------------------ |
| `tearoff`                            | 分窗，0为在原窗，1为点击分为两个窗口 |
| `bg, fg, bd, font, activebackground` |                                      |
| `cursor`                             |                                      |
| `postcommand`                        |                                      |
| `selectcolor`                        | 选中时背景                           |
| `takefocus`                          |                                      |
| `title, type, relief`                |                                      |

方法：
    menu.add_cascade      添加子选项
    menu.add_command      添加命令（label参数为显示内容）
    menu.add_separator    添加分隔线
    menu.add_checkbutton  添加确认按钮
    delete                删除

##### 时间关联`bind`

`bind(sequence,func,add)`
`bind_class(className,sequence,func,add)`
`bind_all(sequence,func,add)`
事件参数：　　
sequence      　　　　　　　　所绑定的事件；
func            　　　　　　 所绑定的事件处理函数；
add             　　　　　　 可选参数，为空字符或‘+’；
className    　　　　　　　 　所绑定的类；

鼠标键盘事件
    <Button-1>        　  　鼠标左键按下，2表示中键，3表示右键；
    <ButtonPress-1>    　   同上；
    <ButtonRelease-1>　　　 鼠标左键释放；
    <B1-Motion>  　　       按住鼠标左键移动；
    <Double-Button-1>  　　 双击左键；
    <Enter>       　　      鼠标指针进入某一组件区域；
    <Leave>    　　         鼠标指针离开某一组件区域；
    <MouseWheel>  　   　　 滚动滚轮；
    <KeyPress-A> 　　  　　  按下A键，A可用其他键替代；
    <Alt-KeyPress-A>　　　   同时按下alt和A；alt可用ctrl和shift替代；
    <Double-KeyPress-A>　　  快速按两下A；
    <Lock-KeyPress-A>　　　  大写状态下按A；

窗口事件
    Activate        　　　　 当组件由不可用转为可用时触发；
    Configure      　　　　  当组件大小改变时触发；
    Deactivate    　　　　　 当组件由可用转变为不可用时触发；
    Destroy        　　　　  当组件被销毁时触发；
    Expose         　　　　　当组件从被遮挡状态中暴露出来时触发；
    Unmap        　　　　　　当组件由显示状态变为隐藏状态时触发；
    Map         　　　　     当组件由隐藏状态变为显示状态时触发；
    FocusIn       　　　 　  当组件获得焦点时触发；
    FocusOut      　　　　　 当组件失去焦点时触发；
    Property     　　　　    当窗体的属性被删除或改变时触发；
    Visibility       　　　　当组件变为可视状态时触发；

响应事件
event对象（def function(event)）：
    char        　　　　　　  按键字符，仅对键盘事件有效；
    keycode   　　　　　　  　按键名，仅对键盘事件有效；
    keysym     　　　　　　　 按键编码，仅对键盘事件有效；
    num          　　　　　　鼠标按键，仅对鼠标事件有效；
    type         　　　　    所触发的事件类型；
    widget      　　　　     引起事件的组件；
    width,heigh　　　　　　  组件改变后的大小，仅Configure有效；
    x,y       　  　　　　　　鼠标当前位置，相对于窗口；
    x_root,y_root　　　　　　 鼠标当前位置，相对于整个屏幕

12、弹窗

messagebox._show函数的控制参数：
    default         指定消息框按钮；
    icon            指定消息框图标；
    message     　 　指定消息框所显示的消息；
    parent          指定消息框的父组件；
    title           标题；
    type            类型；

simpledialog模块参数：
    title           指定对话框的标题；
    prompt        　显示的文字；
    initialvalue    指定输入框的初始值；

　　filedialog　　　　模块参数：
    filetype   　　  指定文件类型；
    initialdir 　　  指定默认目录；
    initialfile 　　 指定默认文件；
    title    　　　  指定对话框标题

colorchooser模块参数：
    initialcolor  　 指定初始化颜色；
    title          　指定对话框标题；

13、字体（font)
一般格式：
（'Times -10 bold')
('Times',10,'bold','italic')    依次表示字体、字号、加粗、倾斜

补充：
config            重新配置
label.config(font='Arial -%d bold' % scale.get())
依次为字体，大小（大小可为字号大小），加粗
tkinter.StringVar    能自动刷新的字符串变量，可用set和get方法进行传值和取值，类似的还有IntVar,DoubleVar...
sys.stdout.flush()　　刷新输出