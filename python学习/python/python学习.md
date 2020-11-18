| 错误                | 原因                                 |
| ------------------- | ------------------------------------ |
| `SyntaxError`       | 语法错误                             |
| `IndentationError`  | 错误的使用缩进量                     |
| `TypeError`         | 类型错误，类型不支持的操作           |
| `NameError`         | 变量或者函数名拼写错误               |
| `IndexError`        | 引用超过list最大索引                 |
| `KeyError`          | 使用不存在的字典键值                 |
| `UnboundLocalError` | 在定义局部变量前在函数中使用局部变量 |

### python标准库

Python的多态是`x.method`的方法运行时，`method`的意义取决于`x`的类型，属性总是在运行期解析

Python允许执行连续比较，且比较链可以任意长：`a<b<c`结果等同于`a<b and b<c`、`a<b>c`结果等同于`a<b and b>c`

比较操作时，Python能够自动遍历嵌套的对象，从左到右递归比较，要多深有多深。过充中首次发现的差异将决定比较的结果。 

变量名由：下划线或字母开头，后面接任意字母、数字、下划线，以单下划线开头的变量名不会被`from module import *`语句导入，如变量名`_x`.

扩展的序列解包赋值：收集右侧值序列中未赋值的项为一个列表，将该列表赋值给带星号`*`的变量

左边的变量名序列长度不需要与值序列的长度相等，其中只能有一个变量名带星号`*`

* 若带星号`*`变量名只匹配一项，则也是产生一个列表，列表中只有一个元素，如`a,*b="12"`，`b`为`[2]`
* 若带星号`*`变量名没有匹配项，则也是产生空列表，如`a,*b="1"`，`b`为`[]`

* 带星号`*`的变量名可以出现在变量名序列中的任何位置如`*a,b="1234"`，`a`为`[1,2,3]`
* 匹配过程优先考虑不带星号的变量名，剩下的才匹配带星号的变量名
* 以下情况会引发错误：

#### 内置函数

| 函数                                              | 描述                                                         |
| ------------------------------------------------- | ------------------------------------------------------------ |
| `ascii(object)`                                   | 返回一个对象可打印的字符串                                   |
| `breakpoint(*arg, **kws)`                         | 此函数会在调用时将你陷入调试器中。具体来说，它调用 `sys.breakpointhook()` ，直接传递 `args` 和 `kws` |
| `class bytearray([source[, encoding,[,errors]]])` | 返回一个新的 bytes 数组，是一个可变序列                      |
| `class bytes([source[, encoding[,errors]]])`      | 返回一个新的“bytes”对象， 是一个不可变序列。                 |
| `callable(object)`                                | `如果实参` object `是可调用的，返回`True`，否则返回 `False`  |
| `chr(i)`                                          | 返回` Unicode `码位为整数` *i* `的字符的字符串格式。         |
| `@classmethod`                                    | 把一个方法封装成类方法。                                     |
| `dir([object])`                                   | 如果没有实参，则返回当前本地作用域中的名称列表。如果有实参，它会尝试返回该对象的有效属性列表。 |
| `enumerate(iterable, start = 0)`                  | 返回一个枚举对象。`iterable`必须是一个序列，或 `iterator`，或其他支持迭代的对象。 |
| `eval(expression, globals=None, locals=None)`     | 实参是一个字符串，以及可选的` globals `和` locals`。`globals`实参必须是一个字典。`locals`可以是任何映射对象。 |
| `filter(function, iterable)`                      | 用`iterable`中函数`function `返回真的那些元素，构建一个新的迭代器。 |
| `getattr(object,name[,default])`                  | 返回对象命名属性的值。`name`必须是字符串。如果该字符串是对象的属性之一，则返回该属性的值。 |
| `globals()`                                       | 返回表示当前全局符号表的字典。                               |
| `hasattr(object,name)`                            | 该实参是一个对象和一个字符串。如果字符串是对象的属性之一的名称，则返回 `True`，否则返回 `False`。 |
| `hash(object)`                                    | 返回该对象的哈希值                                           |
| `id(object)`                                      | 返回对象的“标识值”。该值是一个整数，在此对象的生命周期中保证是唯一且恒定的。 |
| `isinstance(object,classinfo)`                    | 如果` object `实参是` classinfo `实参的实例，或者是（直接、间接或 虚拟）子类的实例，则返回`true`。如果` object `不是给定类型的对象，函数始终返回` false`。如果` classinfo `是对象类型（或多个递归元组）的元组，如果` object `是其中的任何一个的实例则返回` true`。 |
| `issubclass(class, classinfo)`                    | 如果` class `是` classinfo `的子类--直接、间接或 虚拟 的，则返回` true`。`classinfo `可以是类对象的元组，此时` classinfo `中的每个元素都会被检查。 |
| `iter(object[,sentinel])`                         | 返回一个iterator对象。                                       |
| `locals()`                                        | 更新并返回表示当前本地符号表的字典。                         |
| `map(function, iterable,...)`                     | 产生一个将 *function* 应用于迭代器中所有元素并返回结果的迭代器。 |
| `next(iterator[, default])`                       | 通过调用`iterator`的`__next__()`方法获取下一个元素。如果迭代器耗尽，则返回给定的`default`，如果没有默认值则触发`StopIteration` |
| `open(file, mode='r', encoding=None)`             | file 是一个path-like object，表示将要打开的文件的路径。mode 是一个可选字符串，用于指定打开文件的模式。encoding 是用于解码或编码文件的编码的名称。errors 是一个可选的字符串参数，用于指定如何处理编码和解码错误。 |
| `class property(fget=None, fset=None, fdel=None)` | 返回 property 属性。fget是获取属性值的函数。 *fset* 是用于设置属性值的函数。 *fdel* 是用于删除属性值的函数。并且 *doc*为属性对象创建文档字符串。 |
| `range(start, stop[, step])`                      |                                                              |
| `reversed(seq)`                                   | 返回一个反向的`iterator`。 `seq`必须是一个具有`__reversed__()`方法的对象或者是支持该序列协议 |
| `setattr(object, name, value)`                    | 其参数为一个对象、一个字符串和一个任意值。 字符串指定一个现有属性或者新增属性。 函数会将值赋给该属性，只要对象允许这种操作。 |
| `class slice(start, stop[,step])`                 | 返回一个表示由`range(start, stop, step)`所指定索引集的`slice`对象。 其中start 和 step 参数默认为None。 切片对象具有仅会返回对应参数值的只读数据属性 start, stop 和 step。 |
| `sorted(iterable,*, key =None, reversed=False)`   |                                                              |
| `@staticmethod`                                   | 将方法转换为静态方法。                                       |
| `super([type[, object-or-type]])`                 |                                                              |
| `type(name, bases, dict)`                         |                                                              |
| `vars([object])`                                  | 返回模块、类、实例或任何其它具有`__dict__` 属性的对象的` __dict__` 属性。 |
| `zip(*iterables)`                                 |                                                              |

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

| 方法                                      | 描述                                                         |
| ----------------------------------------- | ------------------------------------------------------------ |
| `@lru_cache(maxsize=128, typed=False)`    | 一个提供缓存功能的装饰器，包装一个函数，缓存其`maxsize`组传入参数，在下次以相同参数调用时直接返回上一次的结果。用以节约高开销或I/O函数的调用时间。 |
| `partial(func, *args, **keywords)`        | The`partial()`is used for partial function application which "freezes" some portion of a function's arguments and/or keywords resulting in a new object with a simplified signature. |
| `reduce(function,iterable[,initializer])` | Apply *function* of two arguments cumulatively to the items of *sequence*, from left to right, so as to reduce the sequence to a single value. |

##### `operator`

| 方法                           | 描述                                                         |
| ------------------------------ | ------------------------------------------------------------ |
| `getitem(a,b)`                 | Return the value of *a* at index *b*.                        |
| `setitem(a,b,c)`               | Set the value of *a* at index *b* to *c*.                    |
| `attrgetter(*attrs)`           | Return a callable object that fetches `attr`from its operand. After` f = attrgetter('name')`, the call`f(b)` returns `b.name`. After` f = attrgetter('name', 'date')`, the call `f(b) `returns` (b.name, b.date)`. |
| `itemgetter(*item)`            | After `f = itemgetter(2)`, the call `f(r)` returns `r[2]`. After `g = itemgetter(2, 5, 3)`, the call `g(r)` returns `(r[2], r[5], r[3])`. |
| `methodcaller(name[,args...])` | After `f = methodcaller('name')`, the call `f(b)` returns `b.name()`. After `f = methodcaller('name', 'foo', bar=1)`, the call `f(b)` returns `b.name('foo', bar=1)`. |
| `countOf(a,b)`                 | Return the number of occurrences of *b* in *a*.              |
| `indexOf(a,b)`                 | Return the index of the first of occurrence of *b* in *a*.   |
| `concat(a,b)`                  | Return `a + b` for *a* and *b* sequences.                    |

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

| 方法和属性                                       | 描述                                                         |
| ------------------------------------------------ | ------------------------------------------------------------ |
| `PurePath.parts`                                 | 一个元组，可以访问路径的多个组件                             |
| `drive`                                          | 一个表示驱动器盘符或命名的字符串，如果存在                   |
| `parents`                                        | An immutable sequence providing access to the logical ancestors of the path |
| `parent`                                         | 此路径的逻辑父路径                                           |
| `name`                                           | 一个表示最后路径组件的字符串，排除了驱动器与根目录           |
| `suffix`                                         | 最后一个组件的文件扩展名，                                   |
| `stem`                                           | 最后一个路径组件，除去后缀                                   |
| `as_posix()`                                     | 返回使用正斜杠`/`的路径字符串                                |
| `is_absolute()`                                  | 返回此路径是否为绝对路径。                                   |
| `joinpath(*other)`                               | 调用此方法等同于将每个 *other* 参数中的项目连接在一起        |
| `match(pattern)`                                 | 将此路径与提供的通配符风格的模式匹配。如果匹配成功则返回 `True`，否则返回 `False`。 |
| `with_name(name)`                                | 返回一个新的路径并修改 name。                                |
| `with_suffix(suffix)`                            | 返回一个新的路径并修改 suffix。                              |
| `Path.cwd()`                                     | 返回一个新的表示当前目录的路径对象                           |
| `home()`                                         | 返回一个表示当前用户家目录的新路径对象                       |
| `stat`                                           | 返回此路径的信息                                             |
| `chmod(mod)`                                     | 改变文件的模式和权限                                         |
| `exists()`                                       | 此路径是否指向一个已存在的文件或目录                         |
| `is_dir()`                                       | 如果路径指向一个目录则返回 `True`，如果指向其他类型的文件则返回 `False`。 |
| `is_file()`                                      | 如果路径指向一个正常的文件则返回 `True`，如果指向其他类型的文件则返回 `False`。 |
| `iterdir()`                                      | 当路径指向一个目录时，产生该路径下的对象的路径               |
| `mkdir(mode=0o777,parents=False,exist_ok=False)` | 新建给定路径的目录。如果给出了 *mode* ，它将与当前进程的 `umask` 值合并来决定文件模式和访问标志。 |
| `open(mode='r', encoding=None)`                  | 打开路径指向的文件，就像内置的 open() 函数所做的一样         |
| `owener()`                                       | 返回拥有此文件的用户名。                                     |
| `read_text(encoding=None, errors=None)`          | 以字符串形式返回路径指向的文件的解码后文本内容               |
| `rename(target)`                                 | 使用给定的 *target* 将文件重命名。                           |
| `replace(target)`                                | 使用给定的 *target* 重命名文件或目录。如果 *target* 指向现存的文件或目录，则将被无条件覆盖。 |
| `resolve(strict=False)`                          | 将路径绝对化，解析任何符号链接。返回新的路径对象             |
| `rmdir()`                                        | 移除此目录。此目录必须为空的。                               |
| `samefile(other_path)`                           | 返回此目录是否指向与可能是字符串或者另一个路径对象的 *other_path* 相同的文件。 |
| `touch(mode=0o666, exist_ok=True)`               | 将给定的路径创建为文件。如果给出了 *mode* 它将与当前进程的 `umask` 值合并以确定文件的模式和访问标志。 |
| `write_text(data, encoding=None, errors=None)`   | 将文件以文本模式打开，写入 *data* 并关闭                     |

##### `os.path`

| 方法                                | 描述                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| `os.path.abspath(path)`             | Return a normalized $absolutized$ version of the pathname *path*. |
| `basename(path)`                    | Return the base name of pathname *path*.                     |
| `commonprefix(list)`                | Return the longest path prefix that is a prefix of all paths in *list*. |
| `dirname(path)`                     | Return the directory name of pathname *path*.                |
| `exists(path)`                      | Return `True` if *path* refers to an existing path or an open file descriptor |
| `getatime(path), getmtime, getsize` | Return time of last access/time of last modification/the size, in bytes, of *path*. |
| `isabs(path)`                       | Return `True` if *path* is an absolute pathname.             |
| `isfile(path), isdir`               | Return `True` if *path* is an `existing`regular file/directory. |
| `join(path, *paths)`                | Join one or more path components intelligently.              |
| `split(path)`                       | Split the pathname *path* into a pair, `(head, tail)` where *tail* is the last pathname component and *head* is everything leading up to that. |

##### `os`

获取当前工作目录：`os.getcwd`
创建单个目录：`os.mkdir`
创建嵌套目录结构：`os.makedirs`  `os.makedirs('one/two')`
删除目录：`os.rmdir`：不能删除非空目录
删除多个目录：`os.rmdirs`：递归删除树结构中的目录。

##### `sys`

| 函数名                         | 作用                                                         |
| ------------------------------ | ------------------------------------------------------------ |
| `argv`                         |                                                              |
| `getfilessystemencoding()`     | Return the name of the encoding used to convert between Unicode filenames and bytes filenames. |
| `getsizeof(object[, default])` | Return the size of an object in bytes. The object can be any type of object. |
| `sys.path`                     | A list of strings that specifies the search path for modules. Initialized from the environment variable `PYTHONPATH`, plus an installation-dependent default. |



#### Data persistence

##### `pickle`

The pickle module implements binary protocols for serializing and de-serializing a Python object structure

To serialize an object hierarchy, you simply call the `dumps()` function. Similarly, to de-serialize a data stream, you call the `loads()` function. However, if you want more control over serialization and de-serialization, you can create a `Pickler` or an `Unpickler` object, respectively.

| 函数名                | 描述                                                         |
| --------------------- | ------------------------------------------------------------ |
| `dump(obj, file,*)`   | Write a pickled representation of obj to the open file object file. This is equivalent to `Pickler(file, protocol).dump(obj)`. |
| `dumps(obj)`          | Return the pickled representation of the object as a bytes object, instead of writing it to a file. |
| `load(file, *)`       | Read a pickled object representation from the open file object file and return the reconstituted object hierarchy specified therein. This is equivalent to `Unpickler(file).load()`. |
| `loads(bytes_object)` | Read a pickled object hierarchy from a bytes object and return the reconstituted object hierarchy specified therein. |

The following types can be pickled: `None, True`, and `False`; integers, floating point numbers, complex numbers; strings, bytes, `bytearrays`; tuples, lists, sets, and dictionaries containing only `picklable` objects; functions defined at the top level of a module --using def, not lambda; built-in functions defined at the top level of a module; classes that are defined at the top level of a module; instances of such classes whose `__dict__ `or the result of calling `__getstate__() `is `picklable`

#### 数据压缩和存档

##### `zlib`

For applications that require data compression, the functions in this module allow compression and decompression, using the `zlib` library.

| 函数名                            | 作用                                                         |
| --------------------------------- | ------------------------------------------------------------ |
| `zlib.compress(data, level = -1)` | Compresses the bytes in data, returning a bytes object containing compressed data. level is an integer from 0 to 9 or -1 controlling the level of compression |
| `compressobj(level=-1)`           | Returns a compression object, to be used for compressing data streams that won’t fit into memory at once. |
| `decompress(data)`                | Decompresses the bytes in *data*, returning a bytes object containing the uncompressed data. |
| `decompressobj()`                 | Returns a decompression object, to be used for decompressing data streams that won’t fit into memory at once. |
| `Compress.compress(data)`         | Compress *data*, returning a bytes object containing compressed data for at least part of the data in *data*. |

##### `gzip`

This module provides a simple interface to compress and decompress files just like the GNU programs `gzip`and `gunzip` would.

The *mode* argument can be any of `'r'`, `'rb'`, `'a'`, `'ab'`, `'w'`, `'wb'`, `'x'` or `'xb'` for binary mode, or `'rt'`, `'at'`, `'wt'`, or `'xt'` for text mode. The default is `'rb'`.

| 函数名                                                       | 作用                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `gzip.open(filename, mode='rb', encoding=None, compresslevel=9)` | Open a gzip-compressed file in binary or text mode, returning a file object. The filename argument can be an actual filename, or an existing file object to read from or write to. |
| `gzip.GzipFile(同上)`                                        | Constructor for the `GzipFile` class, which simulates most of the methods of a file object, with the exception of the truncate() method. |
| `compress(data, compresslevel=9)`                            | Compress the data, returning a bytes object containing the compressed data. |
| `decompress(data)`                                           | Decompress the data, returning a bytes object containing the uncompressed data. |

##### `zipfile`

The ZIP file format is a common archive and compression standard. This module provides tools to create, read, write, append, and list a ZIP file.

| 函数名                                       | 作用                                                         |
| -------------------------------------------- | ------------------------------------------------------------ |
| `zipfile.ZipFile(file, model='r')`           | Open a ZIP file, where file can be a path to a file (a string), a file-like object or a path-like object. |
| `ZipFile.close()`                            | Close the archive file.                                      |
| `ZipFile.getinfo(name)`                      | Return a `ZipInfo` object with information about the archive member name. |
| `ZipFile.infolist()`                         | Return a list containing a `ZipInfo` object for each member of the archive. |
| `ZipFile.open(name, mode='r', pwd=None)`     | Access a member of the archive as a binary file-like object. name can be either the name of a file within the archive or a `ZipInfo` object. |
| `ZipFile.extract(number, path=None)`         | Extract a member from the archive to the current working directory; member must be its full name or a `ZipInfo` object. |
| `ZipFile.extractall(path=None, number=None)` | Extract all members from the archive to the current working directory. |
| `ZipFile.read(name)`                         | Return the bytes of the file name in the archive. name is the name of the file in the archive, or a `ZipInfo` object. |
| `ZipFile.write(filename,arcname=None)`       | Write the file named *filename* to the archive, giving it the archive name `arcname` by default, this will be the same as *filename*, but without a drive letter and with leading path separators removed |

```python
import tarfile
tar = tarfile.open('sample.tar.gz')
tar.extractall()
tar.close
```

#### 数据格式化

##### `csv`

`csv.reader(csvfile, dialect='excel', **fmtparams)`：Return a reader object which will iterate over lines in the given`csvfile`. `csvfile` can be any object which supports the iterator protocol and returns a string each time its `__next__() `method is called --- file objects and list objects are both suitable. 

`csv.writer(csvfile, dialect='excel', **fmtparams)`：Return a writer object responsible for converting the user's data into delimited strings on the given file-like object. `csvfile` can be any object with a `write()` method. If `csvfile` is a file object, it should be opened with newline=''.

`csv.DictReader(f, fieldname=None, restkey = None, restval=None, dialect='excel', *args, **kwds)`：Create an object that operates like a regular reader but maps the information in each row to an `OrderedDict` whose keys are given by the optional `fieldnames` parameter.

`csv.DictWriter(f, fieldnames, restval='', extrasaction='raise', dialect='excel', *args, **kwds)`：Create an object which operates like a regular writer but maps dictionaries onto output rows. The `fieldnames` parameter is a `sequence`of keys that identify the order in which values in the dictionary passed to the `writerow()`method are written to file *f*. The optional `restval` parameter specifies the value to be written if the dictionary is missing a key in `fieldnames`.

| 方法                      | 描述                                                         |
| ------------------------- | ------------------------------------------------------------ |
| `csvreader.__next__()`    | Return the next row of the reader's `iterable` object as a list or a `dict`, parsed according to the current dialect |
| `line_num`                | The number of lines read from the source iterator.           |
| `fieldnames`              | If not passed as a parameter when creating the object, this attribute is initialized upon first access or when the first record is read from the file. |
| `csvwriter.writerow(row)` | Write the *row* parameter to the writer's file object.       |
| `writerows(rows)`         | Write all elements in *rows*  to the writer's file object.   |

#### 其他

###### pip

  `install`: Install packages.  `download`: Download packages.  `uninstall`: $Uninstall$ packages.  `freeze`                     Output installed packages in requirements format. ` list `: List installed packages. ` show `: Show information about installed packages. ` check `: Verify installed packages have compatible dependencies.                             ` config `: Manage local and global configuration. ` search` : Search $PyPI$ for packages. ` wheel`Build wheels from your requirements. ` hash`: Compute hashes of package archives. ` completion `  A helper command used for command completion.  `help  `: Show help for commands.

General Options:
 ` -h, --help`: Show help.   `-V, --version  `: Show version and exit.  ` --log <path> ` : Path to a verbose appending log.  ` --proxy <proxy> ` : Specify a proxy in the form $[user:passwd@]proxy.server:port$.`--retries <retries>  ` : Maximum number of retries each connection should attempt. `--timeout <sec> `: Set the socket timeout (default 15 seconds).  `--cache-dir <dir>`: Store the cache data in <dir>`--no-cache-dir`: Disable the cache.

`pip install`

 `-r, --requirement <file> `: Install from the given requirements file. This option can be used multiple times.   `-c, --constraint <file>`: Constrain versions using the given constraints file. This option can be used multiple times.  `--no-deps`: Don't install package dependencies.   `--pre`: Include pre-release and development versions. By default, pip only finds stable versions.   `-t, --target <dir>`: Install packages into <dir>. By default this will not replace existing files/folders in<dir>. Use --upgrade to replace existing packages in <dir> with new versions.     `-U, --upgrade`: Upgrade all specified packages to the newest available version. The handling of dependencies depends on the upgrade-strategy used.   `--user ` : Install to the Python user install directory for your platform.   `--root <dir>`: Install everything relative to this alternate root directory.   `-i, --index-url <url>`: Base URL of Python Package Index This should point to
a repository compliant with PEP 503 or a local directory laid out in the same format.