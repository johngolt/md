| 错误                | 原因                                 |
| ------------------- | ------------------------------------ |
| `SyntaxError`       | 语法错误                             |
| `IndentationError`  | 错误的使用缩进量                     |
| `TypeError`         | 类型错误，类型不支持的操作           |
| `NameError`         | 变量或者函数名拼写错误               |
| `IndexError`        | 引用超过list最大索引                 |
| `KeyError`          | 使用不存在的字典键值                 |
| `UnboundLocalError` | 在定义局部变量前在函数中使用局部变量 |

### 入门教程

### python标准库

#### 内置函数

| 函数                                                     | 描述                                                         |
| -------------------------------------------------------- | ------------------------------------------------------------ |
| `abs(x)`                                                 | 返回一个数的绝对值。实参可以是整数或浮点数。如果实参是一个复数，返回它的模。 |
| `all(iterable)`                                          | 如果 `iterable` 的所有元素为真（或迭代器为空），返回 `True`  |
| `any(iterable)`                                          | 如果`iterable`的任一元素为真则返回`True`。如果迭代器为空，返回`False` |
| `ascii(object)`                                          | 返回一个对象可打印的字符串                                   |
| `bin(x)`                                                 | 将一个整数转变为一个前缀为`0b`的二进制字符串。               |
| `class bool([x])`                                        | 返回一个布尔值，`True` 或者 `False`                          |
| `breakpoint(*arg, **kws)`                                | 此函数会在调用时将你陷入调试器中。具体来说，它调用 `sys.breakpointhook()` ，直接传递 `args` 和 `kws` |
| `class bytearray([source[, encoding,[,errors]]])`        | 返回一个新的 bytes 数组，是一个可变序列                      |
| `class bytes([source[, encoding[,errors]]])`             | 返回一个新的“bytes”对象， 是一个不可变序列。                 |
| `callable(object)`                                       | `如果实参` object `是可调用的，返回`True`，否则返回 `False`  |
| `chr(i)`                                                 | 返回` Unicode `码位为整数` *i* `的字符的字符串格式。         |
| `@classmethod`                                           | 把一个方法封装成类方法。                                     |
| `class complex([real[,imag]])`                           | 返回值为` real + imag*1j `的复数，或将字符串或数字转换为复数。 |
| `delattr(obj,name)`                                      | 实参是一个对象和一个字符串。该字符串必须是对象的某个属性。如果对象允许，该函数将删除指定的属性。 |
| `class dict(**kwarg)`                                    | 创建一个新的字典。                                           |
| `dir([object])`                                          | 如果没有实参，则返回当前本地作用域中的名称列表。如果有实参，它会尝试返回该对象的有效属性列表。 |
| `divmod(a,b)`                                            | 它将两个（非复数）数字作为实参，并在执行整数除法时返回一对商和余数。 |
| `enumerate(iterable, start = 0)`                         | 返回一个枚举对象。`iterable`必须是一个序列，或 `iterator`，或其他支持迭代的对象。 |
| `eval(expression, globals=None, locals=None)`            | 实参是一个字符串，以及可选的` globals `和` locals`。`globals`实参必须是一个字典。`locals`可以是任何映射对象。 |
| `filter(function, iterable)`                             | 用`iterable`中函数`function `返回真的那些元素，构建一个新的迭代器。 |
| `format(value[, format_spec])`                           | 将` value `转换为` format_spec `控制的格式化表示。           |
| `class frozenset([iterable])`                            | 返回一个新的`frozenset`对象，它包含可选参数`iterable`中的元素。 `frozenset`是一个内置的类。 |
| `getattr(object,name[,default])`                         | 返回对象命名属性的值。`name`必须是字符串。如果该字符串是对象的属性之一，则返回该属性的值。 |
| `globals()`                                              | 返回表示当前全局符号表的字典。                               |
| `hasattr(object,name)`                                   | 该实参是一个对象和一个字符串。如果字符串是对象的属性之一的名称，则返回 `True`，否则返回 `False`。 |
| `hash(object)`                                           | 返回该对象的哈希值                                           |
| `help([object])`                                         | 启动内置的帮助系统                                           |
| `hex(x)`                                                 | 将整数转换为以`0x`为前缀的小写十六进制字符串。               |
| `id(object)`                                             | 返回对象的“标识值”。该值是一个整数，在此对象的生命周期中保证是唯一且恒定的。 |
| `input([prompt])`                                        | 如果存在 *prompt* 实参，则将其写入标准输出，末尾不带换行符。 |
| `isinstance(object,classinfo)`                           | 如果` object `实参是` classinfo `实参的实例，或者是（直接、间接或 虚拟）子类的实例，则返回`true`。如果` object `不是给定类型的对象，函数始终返回` false`。如果` classinfo `是对象类型（或多个递归元组）的元组，如果` object `是其中的任何一个的实例则返回` true`。 |
| `issubclass(class, classinfo)`                           | 如果` class `是` classinfo `的子类--直接、间接或 虚拟 的，则返回` true`。`classinfo `可以是类对象的元组，此时` classinfo `中的每个元素都会被检查。 |
| `iter(object[,sentinel])`                                | 返回一个iterator对象。                                       |
| `locals()`                                               | 更新并返回表示当前本地符号表的字典。                         |
| `map(function, iterable,...)`                            | 产生一个将 *function* 应用于迭代器中所有元素并返回结果的迭代器。 |
| `max(iterable,*[,key, default])`                         |                                                              |
| `next(iterator[, default])`                              | 通过调用`iterator`的`__next__()`方法获取下一个元素。如果迭代器耗尽，则返回给定的`default`，如果没有默认值则触发`StopIteration` |
| `oct(x)`                                                 | 将一个整数转变为一个前缀为`0o`的八进制字符串。               |
| `open(file, mode='r', encoding=None)`                    | file 是一个path-like object，表示将要打开的文件的路径。mode 是一个可选字符串，用于指定打开文件的模式。encoding 是用于解码或编码文件的编码的名称。errors 是一个可选的字符串参数，用于指定如何处理编码和解码错误。 |
| `ord(c)`                                                 | 对表示单个 Unicode 字符的字符串，返回代表它 Unicode 码点的整数。 |
| `pow(x,y[,z])`                                           | 返回 *x* 的 *y* 次幂；如果 *z* 存在，则对 *z* 取余           |
| `print(*object, sep=' ',end='\n',file=sys.stdout)`       | 将 *objects* 打印到 *file* 指定的文本流，以 *sep* 分隔并在末尾加上 *end*。 *sep*, *end*, *file* 和 *flush* 如果存在，它们必须以关键字参数的形式给出。 |
| `class property(fget=None, fset=None, fdel=None)`        | 返回 property 属性。fget是获取属性值的函数。 *fset* 是用于设置属性值的函数。 *fdel* 是用于删除属性值的函数。并且 *doc*为属性对象创建文档字符串。 |
| `range(start, stop[, step])`                             |                                                              |
| `reversed(seq)`                                          | 返回一个反向的`iterator`。 `seq`必须是一个具有`__reversed__()`方法的对象或者是支持该序列协议 |
| `round(number[, ndigits])`                               | 返回number 舍入到小数点后`ndigits`位精度的值。 如果`ndigits`被省略或为`None`，则返回最接近输入值的整数。 |
| `setattr(object, name, value)`                           | 其参数为一个对象、一个字符串和一个任意值。 字符串指定一个现有属性或者新增属性。 函数会将值赋给该属性，只要对象允许这种操作。 |
| `class slice(start, stop[,step])`                        | 返回一个表示由`range(start, stop, step)`所指定索引集的`slice`对象。 其中start 和 step 参数默认为None。 切片对象具有仅会返回对应参数值的只读数据属性 start, stop 和 step。 |
| `sorted(iterable,*, key =None, reversed=False)`          |                                                              |
| `@staticmethod`                                          | 将方法转换为静态方法。                                       |
| `class str(object=b'', encoding='utf8',errors='strict')` |                                                              |
| `sum(iterable[, start])`                                 | 从 *start* 开始自左向右对`iterable`中的项求和并返回总计值。  |
| `super([type[, object-or-type]])`                        |                                                              |
| `type(name, bases, dict)`                                |                                                              |
| `vars([object])`                                         | 返回模块、类、实例或任何其它具有`__dict__` 属性的对象的` __dict__` 属性。 |
| `zip(*iterables)`                                        |                                                              |
| `__import__(name, globals=None, locals=None,level=0)`    | 此函数会由 import 语句发起调用。                             |

#### 内置类型

The principal built-in types are numerics, sequences, mappings, classes, instances and exceptions. Some collection classes are mutable. The methods that add, subtract, or rearrange their members in place, and don’t return a specific item, never return the collection instance itself but None. Some operations are supported by several object types; in particular, practically all objects can be compared, tested for truth value, and converted to a string

###### 逻辑值检测

一个对象在默认情况下均被视为真值，除非当该对象被调用时其所属类定义了` __bool__()`方法且返回`False`或是定义了`__len__()`方法且返回零。下面基本完整地列出了会被视为假值的内置对象:  被定义为假值的常量: `None` 和`False`。任何数值类型的零:` 0, 0.0, 0j, Decimal(0), Fraction(0, 1)`。空的序列和多项集:` '', (), [], {}, set(), range(0)`。产生布尔值结果的运算和内置函数总是返回`0`或`False`作为假值，`1`或`True`作为真值。

###### 序列类型

所有序列规定的比较操作都是基于字典顺序，即一个元素接一个元素地比较，直至找到第一个不同的元素。大多数序列类型支持下表中的操作，表中s和t是具有相同类型的序列。

| 运算                 | 结果                                        |
| -------------------- | ------------------------------------------- |
| `x in s`             | 如果s中的某项等于x则结果为True，否则为False |
| `x not in s`         | 如果s中某项等于x则结果为False，否则为True   |
| `s +t`               | s与t相拼接                                  |
| `s*n`                |                                             |
| `s[i]`               |                                             |
| `s[i:j]`             |                                             |
| `s[i:j:k]`           |                                             |
| `len(s)`             |                                             |
| `min(s)`             |                                             |
| `max(s)`             |                                             |
| `s.index(x[,i[,j]])` | x再s中首次出现项的索引号                    |
| `s.count(x)`         | x在s中出现的总次数                          |

不可变序列类型普遍实现而可变序列类型未实现的唯一操作就是对`hash()`内置函数的支持。下标是可变序列类型支持的操作。

| 运算            | 结果                                       |
| --------------- | ------------------------------------------ |
| `s[i]=k`        |                                            |
| `s[i:j]=k`      |                                            |
| `del s[i:j]`    |                                            |
| `s[i:j:k]=t`    |                                            |
| `s.append(x)`   |                                            |
| `s.clear()`     |                                            |
| `s.copy()`      |                                            |
| `s.extend(t)`   |                                            |
| `s*=n`          |                                            |
| `s.insert(i,x)` | 在由 *i* 给出的索引位置将 *x*插入 *s*      |
| `s.pop([i])`    | 提取在 *i* 位置上的项，并将其从 *s* 中移除 |
| `s.remove(x)`   | 删除 *s* 中第一个 `s[i]` 等于 *x*的项目。  |
| `s.reverse()`   | 就地将列表中的元素逆序                     |

###### 文本序列类型

字符串是由 Unicode 码位构成的不可变序列。

| 方法                                      | 描述                                                         |
| ----------------------------------------- | ------------------------------------------------------------ |
| `endswith(suffix[, start[,end]])`         | 如果字符串以指定的 *suffix* 结束返回 `True`，否则返回 `False`。 |
| `find(sub[,start[,end]])`                 | 返回子字符串 *sub* 在 `s[start:end]` 切片内被找到的最小索引。 |
| `index(sub[,start[,end]])`                | 类似于find                                                   |
| `isalnum,isdigit,isalpha...`              | 判断字符串是否满足某些条件                                   |
| `replace(old,new[,count])`                | 返回字符串的副本，其中出现的所有子字符串 *old* 都将被替换为 *new*。 如果给出了可选参数 *count*，则只替换前 *count* 次出现。 |
| `split(sep=None, maxsplit=-1)`            | 返回一个由字符串内单词组成的列表，使用 *sep* 作为分隔字符串。 如果给出了 *maxsplit*，则最多进行 *maxsplit* 次拆分 |
| `startswith(prefix[,start[,end]])`        | 如果字符串以指定的 *prefix* 开始则返回 `True`，否则返回 `False`。 |
| `strip([char])`                           | 返回原字符串的副本，移除其中的前导和末尾字符。 *chars* 参数为指定要移除字符的字符串。 如果省略或为 `None`，则 *chars* 参数默认移除空格符。 |
| `partition(sep)`                          | 在 *sep* 首次出现的位置拆分字符串，返回一个 3 元组，其中包含分隔符之前的部分、分隔符本身，以及分隔符之后的部分。 |
| `count(sub[, start[, end]])`              | 统计子字符串出现的次数                                       |
| `encode(encoding='utf8', error='strict')` | 对字符串进行编码，得到`bytes`                                |
| `join(iterable)`                          | 对可迭代对象进行合并                                         |
| `splitlines([keepends])`                  | 对字符串按行切分                                             |

###### 集合类型

set 对象是由具有唯一性的`hashable`对象所组成的无序多项集。 常见的用途包括成员检测、从序列中去除重复项以及数学中的集合类计算，例如交集、并集、差集与对称差集等等。

| 方法                                | 描述                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| `isdisjoint(other)`                 | 如果集合中没有与 *other* 共有的元素则返回 `True`。           |
| `issubset(other),issuperset(other)` | 检测是否集合中的每个元素都在 *other* 之中。                  |
| `union(*others)`                    | 返回一个新集合，其中包含来自原集合以及 others 指定的所有集合中的元素。 |
| `intersection(*others)`             | 返回一个新集合，其中包含原集合以及 others 指定的所有集合中共有的元素。 |
| `diffenrence(*other)`               | 返回一个新集合，其中包含原集合中在 others 指定的其他集合中不存在的元素。 |
| `update(*others)`                   | 更新集合，添加来自 others 中的所有元素。                     |
| `add(elem)`                         | 将元素 *elem* 添加到集合中。                                 |
| `remove(elem),discard(elem),pop()`  | 从集合中移除元素 *elem*。                                    |
| `clear()`                           | 从集合中移除所有元素。                                       |

###### 字典

`mapping `对象会将` hashable `值映射到任意对象。 映射属于可变对象。字典的键 几乎 可以是任何值。 非` hashable `的值，即包含列表、字典或其他可变类型的值不可用作键。

| 方法                         | 描述                                                         |
| ---------------------------- | ------------------------------------------------------------ |
| `get(key[,default])`         | Return the value for *key* if *key* is in the dictionary, else *default*. |
| `pop(key[, default])`        | If `key` is in the dictionary, remove it and return its value, else return `default`. |
| `popitem()`                  | Remove and return a `(key, value)` pair from the dictionary. Pairs are returned in LIFO order. |
| `setdefault(key[, default])` | 如果字典存在键`key`，返回它的值。如果不存在，插入值为`default`的键 `key` ，并返回 `default`。 |
| `update([other])`            | Update the dictionary with the key/value pairs from *other*, overwriting existing keys. Return `None`. |

`update`方法处理参数`other`的方式，是典型的“鸭子类型”。函数首先检查`other`是否有`keys`方法，如果有，那么`update`函数就把它当作映射对象来处理。否则，函数
会退一步，转而把`other`当作包含了键值对`(key, value)`元素的迭代器。Python 里大多数映射类型的构造方法都采用了类似的逻辑，因此你既可以用一个映射对象来新建一个映射对象，也可以用包含`(key, value)`元素的可迭代对象来初始化一个映射对象。

##### Dictionary view objects

The objects returned by `dict.keys(), dict.values()` and `dict.items()` are view objects. They provide a dynamic view on the `dictionary’s` entries, which means that when the dictionary changes, the view reflects these changes. Dictionary views can be iterated over to yield their respective data, and support membership tests. `len(dictview), iter(dictview), x in dictview`

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

#### 文本处理

##### `re`

`re.I`:忽略大小写；`re.M`：多行模型，改变`^`和`$`行为；`re.S`:点任意匹配模式，改变`.`行为；`re.L`;  `re.U`; `re.X`:详细模式。

`\w`----匹配字母数字字符；`\W`----匹配非字母数字字符；`\d`----匹配数字；`\D`----匹配所有非数字；`\s`----匹配一个空格字符；`\S`----匹配出空格字符外的所有字符；`\t`----匹配制表符；`\n`----匹配换行符；`\r`----匹配回测符；`.`----匹配除`\n`外的所有字符；`()`----对正则表达式进行分组，并返回匹配的文本；`a|b`----匹配a或者b；`^`----开始位置；`$`----结束位置；`{m}`----精确匹配m个；`*`----匹配模式的0次或多次出现；`{m,}`----匹配至少m个；`{m,n}`----匹配个数介于m和n之间；`?`----匹配模式的一次或零次出现；`+`----匹配模式的一次或多次出现；`[]`--用于表示一个字符集合；`{m,n}?`---前一个修饰符的非贪婪模式，只匹配尽量少的字符次数；`(?P<name>...)`---类似正则组合，但是匹配到的子串组在外部是通过定义的 `name` 来获取的；`\`----转义字符；

| 方法                                              | 描述                                                         |
| ------------------------------------------------- | ------------------------------------------------------------ |
| `compile(pattern, flags = 0)`                     | 将正则表达式的样式编译为一个 正则表达式对象，可以用于匹配。  |
| `re.A, re.I, re.L,re.M, re.S, re.X`               |                                                              |
| `re.search(pattern, string, flags=0)`             | 扫描整个字符串找到匹配样式的第一个位置，并返回一个相应的匹配对象。如果没有匹配，就返回一个 None |
| `match(pattern, string, flags=0)`                 | 如果string开始的0或者多个字符匹配到了正则表达式样式，就返回一个相应的匹配对象 。 如果没有匹配，就返回 None |
| `fullmatch(pattern, string, flags=0)`             | 如果整个string匹配到正则表达式样式，就返回一个相应的匹配对象 。 否则就返回一个None |
| `split(pattern, string, maxsplit=0, flags=0)`     | 用 *pattern* 分开 *string*                                   |
| `findall(pattern, string, flags=0)`               | 对 *string* 返回一个不重复的 *pattern* 的匹配列表， *string* 从左到右进行扫描，匹配按找到的顺序返回。 |
| `finditer(pattern, string, flags=0)`              | pattern 在 string 里所有的非重复匹配，返回为一个迭代器 iterator 保存了 匹配对象 。 string 从左到右扫描，匹配按顺序排列。 |
| `re.sub(pattern, repl, string, count=0, flags=0)` | 返回通过使用 *repl* 替换在 *string* 最左边非重叠出现的 *pattern* 而获得的字符串。 |
| `re.escape(pattern)`                              | 转义 *pattern* 中的特殊字符。                                |

编译后的正则表达式对象支持一下方法和属性：

| 方法                                   |                                                              |
| -------------------------------------- | ------------------------------------------------------------ |
| `search(string[,pos[,endpos]])`        |                                                              |
| `pattern.match(string[,pos[,endpos]])` |                                                              |
| `pattern.flags`                        | 正则匹配标记。                                               |
| `pattern.groups`                       | 捕获组合的数量。                                             |
| `pattern.groupindex`                   | 映射由 `(?P<id>)` 定义的命名符号组合和数字组合的字典。如果没有符号组，那字典就是空的。 |

匹配对象总是有一个布尔值 `True`。如果没有匹配的话 `match()` 和 `search()`返回 `None` 所以你可以简单的用 `if` 语句来判断是否匹配。匹配对象支持以下方法和属性：

| 方法                                        | 描述                                                         |
| ------------------------------------------- | ------------------------------------------------------------ |
| `Match.expand(template)`                    |                                                              |
| `match.group([group1,...])`                 | 返回一个或者多个匹配的子组。如果只有一个参数，结果就是一个字符串，如果有多个参数，结果就是一个元组，如果没有参数，组1默认到0。 |
| `Match.groups(default=None)`                | 返回一个元组，包含所有匹配的子组，在样式中出现的从1到任意多的组合。 |
| `Match.groupdict(default=None)`             | 返回一个字典，包含了所有的 *命名* 子组。key就是组名。        |
| `Match.start([group]),end,span,pos, endpos` | 返回 *group* 匹配到的字串的开始和结束标号。*group* 默认为0   |

#### 数据类型

##### collections

| 类名                                | 作用                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| `namedtuple(typename,field_name)`   | 创建命名元组子类的工厂函数,返回一个新的元组子类，名为 `typename` 。`field_names`是一个像 `[‘x’, ‘y’]` 一样的字符串序列。 |
| `deque([iterable[,maxlen]])`        | 类似列表的容器，实现了在两端快速添加(append)和弹出(pop)      |
| `ChainMap(*maps)`                   | 类似字典的容器类，将多个映射集合到一个视图里面               |
| `Counter(iterable or mapping)`      | 字典的子类，提供了可哈希对象的计数功能                       |
| `OrderDict`                         | 字典的子类，保存了他们被添加的顺序                           |
| `defaultdict(default_factory[,..])` | 字典的子类，提供了一个工厂函数，为字典查询提供一个默认值     |
| `UserDict`                          | 封装了字典对象，简化了字典子类化                             |
| `UseList`                           | 封装了列表对象，简化了列表子类化                             |
| `UseString`                         | 封装了字符串对象，简化了字符串子类化                         |

一个 `ChainMap `将多个字典或者其他映射组合在一起，创建一个单独的可更新的视图。 如果没有` maps `被指定，就提供一个默认的空字典，这样一个新链至少有一个映射。底层映射被存储在一个列表中。这个列表是公开的，可以通过` maps `属性存取和更新。没有其他的状态。搜索查询底层映射，直到一个键被找到。不同的是，写，更新和删除只操作第一个映射。

一个`Counter `是一个` dict `的子类，用于计数可哈希对象。它是一个集合，元素像字典键(key)一样存储，它们的计数存储为值。计数可以是任何整数值，包括0和负数。

`elements()`:返回一个迭代器，每个元素重复计数的个数。元素顺序是任意的。如果一个元素的计数小于1，就会忽略它。
`most_common([n])`:返回一个列表，提供` n` 个频率最高的元素和计数。 如果是`None`， `most_common() `:返回计数器中的所有元素。`subtract([iterable-or-mapping])`:从迭代对象或映射对象减去元素。像`dict.update()`但是是减去，而不是替换。输入和输出都可以是0或者负数。`update([iterable-or-mapping])`:从 迭代对象 计数元素或者 从另一个 映射对象添加。 像` dict.update() `但是是加上，而不是替换。另外，迭代对象应该是序列元素，而不是一个`(key, value) `对。

`deque`返回一个新的双向队列对象，从左到右初始化 ，从`iterable`数据创建。如果 `iterable `没有指定，新队列为空。如果`maxlen`没有指定或者是` None `，`deques `可以增长到任意长度。否则，`deque`就限定到指定最大长度。一旦限定长度的`deque`满了，当新项加入时，同样数量的项就从另一端弹出
`append(x)`:添加 x 到右端。`appendleft(x)`:添加 x 到左端。`count(x)`:计算`deque`中个数等于 x 的元素。`extend(iterable)`: 扩展`deque`的右侧，通过添加`iterable`参数中的元素。`extendleft(iterable)`: 扩展`deque`的左侧，通过添加`iterable`参数中的元素。`insert(i, x)`:在位置` i` 插入` x `。如果插入会导致一个限长`deque`超出长度` maxlen `的话，就升起一个` IndexError`。`pop()`: 移去并且返回一个元素，`deque`最右侧的那一个。`popleft()`: 移去并且返回一个元素，`deque`最左侧的那一个。`remove(value)`: 移去找到的第一个`value`。 `reverse()`: 将`deque`逆序排列。返回 `None` 。`rotate(n=1)`: 向右循环移动` n `步。 如果` n `是负数，就向左循环。

##### `heapq`

This module provides an implementation of the heap queue algorithm, also known as the priority queue algorithm. The `API` below differs from textbook heap algorithms in two aspects: (a) We use zero-based indexing. This makes the relationship between the index for a node and the indexes for its children slightly less obvious, but is more suitable since Python uses zero-based indexing. (b) Our pop method returns the smallest item, not the largest.

| 函数名                                       | 作用                                                         |
| -------------------------------------------- | ------------------------------------------------------------ |
| `heappush(heap, item)`                       | Push the value *item* onto the *heap*, maintaining the heap invariant. |
| `heappop(heap)`                              | Pop and return the smallest item from the *heap*, maintaining the heap invariant. |
| `heappushpop(heap, item)`                    | Push *item* on the heap, then pop and return the smallest item from the *heap*. |
| `heapify(x)`                                 | Transform list *x* into a heap, in-place, in linear time.    |
| `merge(*iterable, key =None, reverse=False)` | Merge multiple sorted inputs into a single sorted output     |
| `nlargest(n, iterable, key=None)`            | Return a list with the *n* largest elements from the dataset defined by `iterable`. |

#### 数字和数学模块

##### `math`

| 函数名                            | 作用                                                         |
| --------------------------------- | ------------------------------------------------------------ |
| `ceil(x)`                         | return the ceiling of x, the smallest integer greater than x. |
| `fbas(x)`                         | 求绝对值                                                     |
| `isfinite(x), isinf(x), isnan(x)` |                                                              |
| `trunc(x)`                        | return the real value x truncated to an integral             |
| `math.inf, math.nan,`             | 常量                                                         |

##### random

| 方法                                                         | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `random.seed(a=None, version=2)`                             | 初始化随机数生成器。                                         |
| `getstate(), setstate(state)`                                | 返回捕获生成器当前内部状态的对象。state 应该是从之前调用 `getstate() `获得的，并且` setstate()` 将生成器的内部状态恢复到` getstate() `被调用时的状态。 |
| `randrange(start, stop[,step])`                              | 从 `range(start, stop, step)` 返回一个随机选择的元素。 这相当于 `choice(range(start, stop, step))` ，但实际上并没有构建一个 range 对象。 |
| `choice(seq)`                                                | 从非空序列 *seq* 返回一个随机元素。 如果 *seq* 为空，则引发`IndexError` |
| `choices(population, weights=None, *, cum_weights=None, k=1)` | 从population中选择替换，返回大小为 k 的元素列表。 如果 population 为空，则引发 `IndexError`。 |
| `shuffle(x[,random])`                                        | 将序列 *x* 随机打乱位置。可选参数 *random* 是一个0参数函数。 |
| `sample(population, k)`                                      | 返回从总体序列或集合中选择的唯一元素的 *k* 长度列表。 用于无重复的随机抽样。 |

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

  `install`: Install packages.  `download`: Download packages.  `uninstall`: $Uninstall$ packages.  `freeze`                      Output installed packages in requirements format. ` list `: List installed packages. ` show `: Show information about installed packages. ` check `: Verify installed packages have compatible dependencies.                             ` config `: Manage local and global configuration. ` search` : Search $PyPI$ for packages. ` wheel`Build wheels from your requirements. ` hash`: Compute hashes of package archives. ` completion `  A helper command used for command completion.  `help  `: Show help for commands.

General Options:
 ` -h, --help`: Show help.   `-V, --version  `: Show version and exit.  ` --log <path> ` : Path to a verbose appending log.  ` --proxy <proxy> ` : Specify a proxy in the form $[user:passwd@]proxy.server:port$.`--retries <retries>  ` : Maximum number of retries each connection should attempt. `--timeout <sec> `: Set the socket timeout (default 15 seconds).  `--cache-dir <dir>`: Store the cache data in <dir>`--no-cache-dir`: Disable the cache.

`pip install`

 `-r, --requirement <file> `: Install from the given requirements file. This option can be used multiple times.   `-c, --constraint <file>`: Constrain versions using the given constraints file. This option can be used multiple times.  `--no-deps`: Don't install package dependencies.   `--pre`: Include pre-release and development versions. By default, pip only finds stable versions.   `-t, --target <dir>`: Install packages into <dir>. By default this will not replace existing files/folders in<dir>. Use --upgrade to replace existing packages in <dir> with new versions.     `-U, --upgrade`: Upgrade all specified packages to the newest available version. The handling of dependencies depends on the upgrade-strategy used.   `--user ` : Install to the Python user install directory for your platform.   `--root <dir>`: Install everything relative to this alternate root directory.   `-i, --index-url <url>`: Base URL of Python Package Index This should point to
a repository compliant with PEP 503 or a local directory laid out in the same format.