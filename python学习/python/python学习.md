| 错误                  | 原因                 |
| ------------------- | ------------------ |
| `SyntaxError`       | 语法错误               |
| `IndentationError`  | 错误的使用缩进量           |
| `TypeError`         | 类型错误，类型不支持的操作      |
| `NameError`         | 变量或者函数名拼写错误        |
| `IndexError`        | 引用超过list最大索引       |
| `KeyError`          | 使用不存在的字典键值         |
| `UnboundLocalError` | 在定义局部变量前在函数中使用局部变量 |

### python标准库

Python的多态是`x.method`的方法运行时，`method`的意义取决于`x`的类型，属性总是在运行期解析 

变量名由下划线或字母开头，后面接任意字母、数字、下划线，以单下划线开头的变量名不会被`from module import *`语句导入，如变量名`_x`.

扩展的序列解包赋值：收集右侧值序列中未赋值的项为一个列表，将该列表赋值给带星号`*`的变量

左边的变量名序列长度不需要与值序列的长度相等，其中只能有一个变量名带星号`*`

* 若带星号`*`变量名只匹配一项，则也是产生一个列表，列表中只有一个元素，如`a,*b="12"`，`b`为`[2]`

* 若带星号`*`变量名没有匹配项，则也是产生空列表，如`a,*b="1"`，`b`为`[]`

* 带星号`*`的变量名可以出现在变量名序列中的任何位置如`*a,b="1234"`，`a`为`[1,2,3]`

* 匹配过程优先考虑不带星号的变量名，剩下的才匹配带星号的变量名


#### 内置函数

`callable(object)`、`chr(i)`、`enumerate(iterable, start = 0)`、`getattr(object,name[,default])`、`hasattr(object,name)`、`isinstance(object,classinfo)`

`zip(*iterables)`

`super([type[, object-or-type]])`

`sorted(iterable,*, key =None, reversed=False)`

`class slice(start, stop[,step])`

`setattr(object, name, value)`

`next(iterator[, default])`

`type(name, bases, dict)`

`map(function, iterable,...)`

`iter(object[,sentinel])`

#### 内置类型

###### 逻辑值检测

一个对象在默认情况下均被视为真值，除非当该对象被调用时其所属类定义了` __bool__()`方法且返回`False`或是定义了`__len__()`方法且返回零。下面基本完整地列出了会被视为假值的内置对象:  被定义为假值的常量: `None` 和`False`。任何数值类型的零:` 0, 0.0, 0j, Decimal(0), Fraction(0, 1)`。空的序列和多项集:` '', (), [], {}, set(), range(0)`。产生布尔值结果的运算和内置函数总是返回`0`或`False`作为假值，`1`或`True`作为真值。

##### 内置错误

程序可以通过创建新的异常类来命名它们自己的异常。异常通常应该直接或间接地从 `Exception` 类派生。

```python
try:
    run this code
except:
    execute this code when there is an exception
else:
    No exceptions run this code
finally:
    always run this code
```

`try `语句的工作原理如下。首先，执行` try`句。如果没有异常发生，则跳过` except `子句并完成` try `语句的执行。
如果在执行`try `子句时发生了异常，则跳过该子句中剩下的部分。然后，如果异常的类型和` except`关键字后面的异常匹配，则执行` except `子句 ，然后继续执行` try`语句之后的代码。如果发生的异常和` except`子句中指定的异常不匹配，则将其传递到外部的` try `语句中；如果没有找到处理程序，则它是一个未处理异常，执行将停止并显示如上所示的消息。一个` try` 语句可能有多个` except `子句，以指定不同异常的处理程序。 最多会执行一个处理程序。 处理程序只处理相应的` try `子句中发生的异常，而不处理同一` try `语句内其他处理程序中的异常。`try ... except `语句有一个可选的` else `子句，在使用时必须放在所有的` except `子句后面。对于在`try `子句不引发异常时必须执行的代码来说很有用。`finally `子句 总会在离开` try `语句前被执行，无论是否发生了异常。 当在 `try` 子句中发生了异常且尚未被` except `子句处理时，它将在` finally `子句执行后被重新抛出。 当` try `语句的任何其他子句通过` break, continue `或` return `语句离开时，`finally `也会在“离开之前”被执行。

#### 函数式编程模块

##### `functools`

| 方法                                        | 描述                                                                                                                                                                                   |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `@lru_cache(maxsize=128, typed=False)`    | 一个提供缓存功能的装饰器，包装一个函数，缓存其`maxsize`组传入参数，在下次以相同参数调用时直接返回上一次的结果。用以节约高开销或I/O函数的调用时间。                                                                                                      |
| `partial(func, *args, **keywords)`        | The`partial()`is used for partial function application which "freezes" some portion of a function's arguments and/or keywords resulting in a new object with a simplified signature. |
| `reduce(function,iterable[,initializer])` | Apply *function* of two arguments cumulatively to the items of *sequence*, from left to right, so as to reduce the sequence to a single value.                                       |

##### `operator`

| 方法                             | 描述                                                                                                                                                                                                                 |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `getitem(a,b)`                 | Return the value of *a* at index *b*.                                                                                                                                                                              |
| `setitem(a,b,c)`               | Set the value of *a* at index *b* to *c*.                                                                                                                                                                          |
| `attrgetter(*attrs)`           | Return a callable object that fetches `attr`from its operand. After` f = attrgetter('name')`, the call`f(b)` returns `b.name`. After` f = attrgetter('name', 'date')`, the call `f(b) `returns` (b.name, b.date)`. |
| `itemgetter(*item)`            | After `f = itemgetter(2)`, the call `f(r)` returns `r[2]`. After `g = itemgetter(2, 5, 3)`, the call `g(r)` returns `(r[2], r[5], r[3])`.                                                                          |
| `methodcaller(name[,args...])` | After `f = methodcaller('name')`, the call `f(b)` returns `b.name()`. After `f = methodcaller('name', 'foo', bar=1)`, the call `f(b)` returns `b.name('foo', bar=1)`.                                              |
| `countOf(a,b)`                 | Return the number of occurrences of *b* in *a*.                                                                                                                                                                    |
| `indexOf(a,b)`                 | Return the index of the first of occurrence of *b* in *a*.                                                                                                                                                         |
| `concat(a,b)`                  | Return `a + b` for *a* and *b* sequences.                                                                                                                                                                          |

#### 文件和目录访问

##### `pathlib`

该模块提供表示文件系统路径的类，其语义适用于不同的操作系统。路径类被分为提供纯计算操作而没有 I/O 的 纯路径，以及从纯路径继承而来但提供 I/O 操作的 具体路径。

![](../picture/2/42.png)

`pathlib.PurePath(*pathsegments)`：一个通用的类，代表当前系统的路径风格。每一个 `pathsegments`的元素可能是一个代表路径片段的字符串，或者另一个路径对象。当 `pathsegments`为空的时候，假定为当前目录。当给出一些绝对路径，最后一位将被当作锚

斜杠` /`操作符有助于创建子路径，就像` os.path.join() `一样:

```python
from pathlib import PurePath
p = PurePath('/etc')
'C:/'p/'one.py'
```

| 方法和属性                                            | 描述                                                                          |
| ------------------------------------------------ | --------------------------------------------------------------------------- |
| `PurePath.parts`                                 | 一个元组，可以访问路径的多个组件                                                            |
| `drive`                                          | 一个表示驱动器盘符或命名的字符串，如果存在                                                       |
| `parents`                                        | An immutable sequence providing access to the logical ancestors of the path |
| `parent`                                         | 此路径的逻辑父路径                                                                   |
| `name`                                           | 一个表示最后路径组件的字符串，排除了驱动器与根目录                                                   |
| `suffix`                                         | 最后一个组件的文件扩展名，                                                               |
| `stem`                                           | 最后一个路径组件，除去后缀                                                               |
| `as_posix()`                                     | 返回使用正斜杠`/`的路径字符串                                                            |
| `is_absolute()`                                  | 返回此路径是否为绝对路径。                                                               |
| `joinpath(*other)`                               | 调用此方法等同于将每个 *other* 参数中的项目连接在一起                                             |
| `match(pattern)`                                 | 将此路径与提供的通配符风格的模式匹配。如果匹配成功则返回 `True`，否则返回 `False`。                           |
| `with_name(name)`                                | 返回一个新的路径并修改 name。                                                           |
| `with_suffix(suffix)`                            | 返回一个新的路径并修改 suffix。                                                         |
| `Path.cwd()`                                     | 返回一个新的表示当前目录的路径对象                                                           |
| `home()`                                         | 返回一个表示当前用户家目录的新路径对象                                                         |
| `stat`                                           | 返回此路径的信息                                                                    |
| `chmod(mod)`                                     | 改变文件的模式和权限                                                                  |
| `exists()`                                       | 此路径是否指向一个已存在的文件或目录                                                          |
| `is_dir()`                                       | 如果路径指向一个目录则返回 `True`，如果指向其他类型的文件则返回 `False`。                                |
| `is_file()`                                      | 如果路径指向一个正常的文件则返回 `True`，如果指向其他类型的文件则返回 `False`。                             |
| `iterdir()`                                      | 当路径指向一个目录时，产生该路径下的对象的路径                                                     |
| `mkdir(mode=0o777,parents=False,exist_ok=False)` | 新建给定路径的目录。如果给出了 *mode* ，它将与当前进程的 `umask` 值合并来决定文件模式和访问标志。                   |
| `open(mode='r', encoding=None)`                  | 打开路径指向的文件，就像内置的 open() 函数所做的一样                                              |
| `owener()`                                       | 返回拥有此文件的用户名。                                                                |
| `read_text(encoding=None, errors=None)`          | 以字符串形式返回路径指向的文件的解码后文本内容                                                     |
| `rename(target)`                                 | 使用给定的 *target* 将文件重命名。                                                      |
| `replace(target)`                                | 使用给定的 *target* 重命名文件或目录。如果 *target* 指向现存的文件或目录，则将被无条件覆盖。                    |
| `resolve(strict=False)`                          | 将路径绝对化，解析任何符号链接。返回新的路径对象                                                    |
| `rmdir()`                                        | 移除此目录。此目录必须为空的。                                                             |
| `samefile(other_path)`                           | 返回此目录是否指向与可能是字符串或者另一个路径对象的 *other_path* 相同的文件。                              |
| `touch(mode=0o666, exist_ok=True)`               | 将给定的路径创建为文件。如果给出了 *mode* 它将与当前进程的 `umask` 值合并以确定文件的模式和访问标志。                 |
| `write_text(data, encoding=None, errors=None)`   | 将文件以文本模式打开，写入 *data* 并关闭                                                    |

##### `os.path`

| 方法                                  | 描述                                                                                                                                             |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `os.path.abspath(path)`             | Return a normalized $absolutized$ version of the pathname *path*.                                                                              |
| `basename(path)`                    | Return the base name of pathname *path*.                                                                                                       |
| `commonprefix(list)`                | Return the longest path prefix that is a prefix of all paths in *list*.                                                                        |
| `dirname(path)`                     | Return the directory name of pathname *path*.                                                                                                  |
| `exists(path)`                      | Return `True` if *path* refers to an existing path or an open file descriptor                                                                  |
| `getatime(path), getmtime, getsize` | Return time of last access/time of last modification/the size, in bytes, of *path*.                                                            |
| `isabs(path)`                       | Return `True` if *path* is an absolute pathname.                                                                                               |
| `isfile(path), isdir`               | Return `True` if *path* is an `existing`regular file/directory.                                                                                |
| `join(path, *paths)`                | Join one or more path components intelligently.                                                                                                |
| `split(path)`                       | Split the pathname *path* into a pair, `(head, tail)` where *tail* is the last pathname component and *head* is everything leading up to that. |

##### `os`

获取当前工作目录：`os.getcwd`
创建单个目录：`os.mkdir`
创建嵌套目录结构：`os.makedirs`  `os.makedirs('one/two')`
删除目录：`os.rmdir`：不能删除非空目录
删除多个目录：`os.rmdirs`：递归删除树结构中的目录。

##### `sys`

| 函数名                            | 作用                                                                                                                                                            |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `argv`                         |                                                                                                                                                               |
| `getfilessystemencoding()`     | Return the name of the encoding used to convert between Unicode filenames and bytes filenames.                                                                |
| `getsizeof(object[, default])` | Return the size of an object in bytes. The object can be any type of object.                                                                                  |
| `sys.path`                     | A list of strings that specifies the search path for modules. Initialized from the environment variable `PYTHONPATH`, plus an installation-dependent default. |

#### Data persistence

##### `pickle`

The pickle module implements binary protocols for serializing and de-serializing a Python object structure

To serialize an object hierarchy, you simply call the `dumps()` function. Similarly, to de-serialize a data stream, you call the `loads()` function. However, if you want more control over serialization and de-serialization, you can create a `Pickler` or an `Unpickler` object, respectively.

| 函数名                   | 描述                                                                                                                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `dump(obj, file,*)`   | Write a pickled representation of obj to the open file object file. This is equivalent to `Pickler(file, protocol).dump(obj)`.                                                       |
| `dumps(obj)`          | Return the pickled representation of the object as a bytes object, instead of writing it to a file.                                                                                  |
| `load(file, *)`       | Read a pickled object representation from the open file object file and return the reconstituted object hierarchy specified therein. This is equivalent to `Unpickler(file).load()`. |
| `loads(bytes_object)` | Read a pickled object hierarchy from a bytes object and return the reconstituted object hierarchy specified therein.                                                                 |

The following types can be pickled: `None, True`, and `False`; integers, floating point numbers, complex numbers; strings, bytes, `bytearrays`; tuples, lists, sets, and dictionaries containing only `picklable` objects; functions defined at the top level of a module --using def, not lambda; built-in functions defined at the top level of a module; classes that are defined at the top level of a module; instances of such classes whose `__dict__ `or the result of calling `__getstate__() `is `picklable`



#### 数据格式化

##### `csv`

`csv.reader(csvfile, dialect='excel', **fmtparams)`：Return a reader object which will iterate over lines in the given`csvfile`. `csvfile` can be any object which supports the iterator protocol and returns a string each time its `__next__() `method is called --- file objects and list objects are both suitable. 

`csv.writer(csvfile, dialect='excel', **fmtparams)`：Return a writer object responsible for converting the user's data into delimited strings on the given file-like object. `csvfile` can be any object with a `write()` method. If `csvfile` is a file object, it should be opened with newline=''.

`csv.DictReader(f, fieldname=None, restkey = None, restval=None, dialect='excel', *args, **kwds)`：Create an object that operates like a regular reader but maps the information in each row to an `OrderedDict` whose keys are given by the optional `fieldnames` parameter.

`csv.DictWriter(f, fieldnames, restval='', extrasaction='raise', dialect='excel', *args, **kwds)`：Create an object which operates like a regular writer but maps dictionaries onto output rows. The `fieldnames` parameter is a `sequence`of keys that identify the order in which values in the dictionary passed to the `writerow()`method are written to file *f*. The optional `restval` parameter specifies the value to be written if the dictionary is missing a key in `fieldnames`.

| 方法                        | 描述                                                                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `csvreader.__next__()`    | Return the next row of the reader's `iterable` object as a list or a `dict`, parsed according to the current dialect                                   |
| `line_num`                | The number of lines read from the source iterator.                                                                                                     |
| `fieldnames`              | If not passed as a parameter when creating the object, this attribute is initialized upon first access or when the first record is read from the file. |
| `csvwriter.writerow(row)` | Write the *row* parameter to the writer's file object.                                                                                                 |
| `writerows(rows)`         | Write all elements in *rows*  to the writer's file object.                                                                                             |

#### 其他

###### `pip`

```shell
#查询当前环境安装的所有软件包
pip list
#查询 pypi 上含有某名字的包
pip search pkg
#查询当前环境中可升级的包
pip list --outdated
#查询一个包的详细内容
pip show pkg
#在不安装软件包的情况下下载软件包到本地
pip download --destination-directory /local/wheels -r requirements.txt
#可以指定这个目录中安装软件包，而不从 pypi 上安装。
pip install --no-index --find-links=/local/wheels -r requirements.txt
#从下载的包中，自己构建生成 wheel 文件
pip install wheel
pip wheel --wheel-dir=/local/wheels -r requirements.txt
#限定版本进行软件包安装
pip install pkg==2.1.2 # >=, <=
# 导出依赖包列表
pip freeze >requirements.txt
# 从依赖包列表中安装
pip install -r requirements.txt
#指定代理服务器安装
pip install --proxy [user:passwd@]http_server_ip:port pkg
#将其写入配置文件中：$HOME/.config/pip/pip.conf
#卸载软件包
pip uninstall pkg
#升级软件包
pip install --upgrade pkg
```

###### `pip.init`

具体做法：`Win + R` ，输入 `%APPDATA% `在当前目录下新建`pip`文件夹，然后新建`pip.ini`文件，内容如下

```
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host=mirrors.aliyun.com
```

```
豆瓣(douban) http://pypi.douban.com/simple/ 
清华大学 https://pypi.tuna.tsinghua.edu.cn/simple/ 
中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/ 
```
