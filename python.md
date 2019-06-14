###### 异常处理

| 异常类名            | 描述                                              |
| ------------------- | ------------------------------------------------- |
| `Exception`         | ·所有异常类的基类                                 |
| `AttributeError`    | 如果对像`obj`没有`foo`成员，会由语法`obj.foo`引发 |
| `EOFError`          | 一个"end of file" 到达控制台或者文件输入引发错误  |
| `IOError`           | 输入/输出曹祖失败引发错误                         |
| `IndexError`        | 索引超出序列范围引发错误                          |
| `KeyError`          | 请求一个不存在的集合或字典关键字引发错误          |
| `KeyboardInterrupt` | 用户按`Ctrl-C`中断程序引发错误                    |
| `NameError`         | 使用不存在的标识符引发错误                        |
| `StopIteration`     | 下一次遍历的元素不存在时引发错误                  |
| `TypeError`         | 发送给函数的参数类型不正确引发错误                |
| `ValueError`        | 函数参数值非法时引发错误，如`sqrt(-5)`            |
| `ZeroDivisionError` | 除数为0引发错误                                   |

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
如果在执行`try `子句时发生了异常，则跳过该子句中剩下的部分。然后，如果异常的类型和` except`关键字后面的异常匹配，则执行` except `子句 ，然后继续执行` try`语句之后的代码。如果发生的异常和` except`子句中指定的异常不匹配，则将其传递到外部的` try `语句中；如果没有找到处理程序，则它是一个未处理异常，执行将停止并显示如上所示的消息。一个` try` 语句可能有多个` except `子句，以指定不同异常的处理程序。 最多会执行一个处理程序。 处理程序只处理相应的` try `子句中发生的异常，而不处理同一` try `语句内其他处理程序中的异常。`try ... except `语句有一个可选的` else `子句，在使用时必须放在所有的` except `子句后面。对于在`try `子句不引发异常时必须执行的代码来说很有用。`finally `子句 总会在离开` try `语句前被执行，无论是否发生了异常。 当在 `try` 子句中发生了异常且尚未被` except `子句处理时，它将在` finally `子句执行后被重新抛出。 当` try `语句的任何其他子句通过` break, continue `或` return `语句离开时，`finally `也会在“离开之前”被执行，



##### 数据库

###### `SQLite`

$\text{SQLite}$是一种嵌入式数据库，它的数据库就是一个文件。由于$\text{SQLite}$本身是C写的，而且体积很小，所以，经常被集成到各种应用程序中。Python就内置了$\text{SQLite3}$，所以，在Python中使用$\text{SQLite}$，不需要安装任何东西，直接使用。在使用$\text{SQLite}$前，我们先要搞清楚几个概念：表是数据库中存放关系数据的集合，一个数据库里面通常都包含多个表。表和表之间通过外键关联。要操作关系数据库，首先需要连接到数据库，一个数据库连接称为`Connection`；连接到数据库后，需要打开游标，称之为`Cursor`，通过`Cursor`执行语句，然后$\text{SQL}$，获得执行结果。Python定义了一套操作数据库的接口，任何数$\text{API}$据库要连接到Python，只需要提供符合Python标准的数据库驱动即可。由于的$\text{SQLite}$驱动内置在Python标准库中，所以我们可以直接来操作数据$\text{SQLite}$库。使用`Cursor`对象执行`insert`，`update`，`delete`语句时，执行结果由`rowcount`返回影响的行数，就可以拿到执行结果。使用`Cursor`对象执行`select`语句时，通过`featchall()`可以拿到结果集。结果集是一个list，每个元素都是一个tuple，对应一行记录。

###### `MongoDB`

```python
import datetime
import pymongo
from pymongo import MongoClient
client = MongoClient()
client = MongoClient('localhost', 27017)
db = client['test-database']#getting a database
collection = db['test-collection']#Getting a collection
post = {"author": "Mike",
        "text": "My first blog post!",
        "tags": ["mongodb", "python", "pymongo"],
        "date": datetime.datetime.utcnow()}
posts = db.posts
post_id = posts.insert_one(post).inserted_id
posts.find_one({"author": "Eliot"})
result = posts.insert_many(new_posts)
posts.find()#query for more than one document
posts.find({"author": "Mike"})
posts.find({"author": "Mike"}).count()
d = datetime.datetime(2009, 11, 12, 12)
posts.find({"date": {"$lt": d}}).sort("author")
'''Adding indexes can help accelerate certain queries and can also add additional 
functionality to querying and storing documents. '''
result = db.profiles.create_index([('user_id', pymongo.ASCENDING)],
                                   unique=True)
```

##### `SQLAlchemy`

首先是连接到数据库，`SQLALchemy`支持多个数据库引擎，不同的数据库引擎连接字符串不一样，常用的有

```
mysql://username:password@hostname/database
postgresql://username:password@hostname/database
sqlite:////absolute/path/to/database
sqlite:///c:/absolute/path/to/database
```

使用传统的`connection`的方式连接和操作数据库

```python
from sqlalchemy import create_engine

# 数据库连接字符串
DB_CONNECT_STRING = 'sqlite:///:memory:'

# 创建数据库引擎,echo为True,会打印所有的sql语句
engine = create_engine(DB_CONNECT_STRING, echo=True)

# 创建一个connection，这里的使用方式与python自带的sqlite的使用方式类似
with engine.connect() as con:
    # 执行sql语句，如果是增删改，则直接生效，不需要commit
    rs = con.execute('SELECT 5')
    data = rs.fetchone()[0]
    print "Data: %s" % data
```

使用事务可以进行批量提交和回滚

```python
with engine.connect() as connection:
    trans = connection.begin()
    try:
        r1 = connection.execute("select * from User")
        r2 = connection.execute("insert into User(name, age) values(?, ?)", 'bomo', 24)
        trans.commit()
    except:
        trans.rollback()
        raise
```

`connection`是一般使用数据库的方式，`sqlalchemy`还提供了另一种操作数据库的方式，通过`session`对象，`session`可以记录和跟踪数据的改变，在适当的时候提交，并且支持强大的`ORM`的功能，下面是基本使用

```python
from sqlalchemy.orm import sessionmaker
# 创建会话类
DB_Session = sessionmaker(bind=engine)

# 创建会话对象
session = DB_Session()
# dosomething with session
# 用完记得关闭，也可以用with
session.close()
```

上面创建了一个`session`对象，接下来可以操作数据库了，`session`也支持通过`sql`语句操作数据库

```python
session.execute('select * from User')
session.execute("insert into User(name, age) values('bomo', 13)")
session.execute("insert into User(name, age) values(:name, :age)", {'name': 'bomo', 'age':12})
# 如果是增删改，需要commit
session.commit()
```

`ORM`框架，我们先定义两个模型类`User`和`Role`，`sqlalchemy`的模型类继承自一个由`declarative_base()`方法生成的类，我们先定义一个模块`Models.py`生成`Base`类

```python
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
```

```python
from sqlalchemy import Column, Integer, String
from Models import Base

class User(Base):
    __tablename__ = 'User'
    id = Column('id', Integer, primary_key=True, autoincrement=True)
    name = Column('name', String(50))
    age = Column('age', Integer)
```

```python
from sqlalchemy import Column, Integer, String
from Models import Base

class Role(Base):
    __tablename__ = 'Role'
    id = Column('id', Integer, primary_key=True, autoincrement=True)
    name = Column('name', String(50))
```

从上面很容易看出来，这里的模型对应数据库中的表，模型支持的类型有`Integer`, `String`, `Boolean`, `Date`, `DateTime`, `Float`。`Column`构造函数相关设置: `name`：名称；`type_`：列类型；`autoincrement`：自增；`default`：默认值；`index`：索引；`nullable`：可空；`primary_key`：外键

接下来通过session进行增删改查

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from User import User
from Role import Role
from Models import Base

DB_CONNECT_STRING = 'sqlite:////Users/zhengxiankai/Desktop/Document/db.sqlite'
engine = create_engine(DB_CONNECT_STRING, echo=True)
DB_Session = sessionmaker(bind=engine)
session = DB_Session()

# 1. 创建表（如果表已经存在，则不会创建）
Base.metadata.create_all(engine)

# 2. 插入数据
u = User(name = 'tobi', age = 200)
r = Role(name = 'user')

# 2.1 使用add，如果已经存在，会报错
session.add(u)
session.add(r)
session.commit()
print r.id

# 3 修改数据
# 3.1 使用merge方法，如果存在则修改，如果不存在则插入（只判断主键，不判断unique列）
r.name = 'admin'
session.merge(r)

# 3.2 也可以通过这种方式修改
session.query(Role).filter(Role.id == 1).update({'name': 'admin'})

# 4. 删除数据
session.query(Role).filter(Role.id == 1).delete()

# 5. 查询数据
# 5.1 返回结果集的第二项
user = session.query(User).get(2)

# 5.2 返回结果集中的第2-3项
users = session.query(User)[1:3]

# 5.3 查询条件
user = session.query(User).filter(User.id < 6).first()

# 5.4 排序
users = session.query(User).order_by(User.name)

# 5.5 降序（需要导入desc方法）
from sqlalchemy import desc
users = session.query(User).order_by(desc(User.name))

# 5.6 只查询部分属性
users = session.query(User.name).order_by(desc(User.name))
for user in users:
    print user.name

# 5.7 给结果集的列取别名
users = session.query(User.name.label('user_name')).all()
for user in users:
    print user.user_name

# 5.8 去重查询（需要导入distinct方法）
from sqlalchemy import distinct
users = session.query(distinct(User.name).label('name')).all()

# 5.9 统计查询
user_count = session.query(User.name).order_by(User.name).count()
age_avg = session.query(func.avg(User.age)).first()
age_sum = session.query(func.sum(User.age)).first()

# 5.10 分组查询
users = session.query(func.count(User.name).label('count'), User.age).group_by(User.age)
for user in users:
    print 'age:{0}, count:{1}'.format(user.age, user.count)

# 6.1 exists查询(不存在则为~exists())
from sqlalchemy.sql import exists
session.query(User.name).filter(~exists().where(User.role_id == Role.id))
# SELECT name AS users_name FROM users WHERE NOT EXISTS (SELECT * FROM roles WHERE users.role_id = roles.id)

# 6.2 除了exists，any也可以表示EXISTS
session.query(Role).filter(Role.users.any())

# 7 random
from sqlalchemy.sql.functions import random
user = session.query(User).order_by(random()).first()

session.close()
```

上面的所有操作都是基于单个表的操作，下面是多表以及关系的使用，我们修改上面两个表，添加外键关联一对多和多对一

##### io编程

在python中访问文件要先调用一个内置函数`open`，它返回一个与底层文件交互的对象。在处理一个文件时，文件对象使用距离文件开始处的偏移量维护文件中的当前位置。在以只读权限或只写权限打开文件时，初始位置是0；如果以追加权限打开，初始位置是在文件的末尾。`fp = open('sample.txt')`

| 调用方法             | 描述                                                         |
| -------------------- | ------------------------------------------------------------ |
| `fp.read()`          | 将只读文件剩下的所有内容作为一个字符串返回                   |
| `fp.read(k)`         | 将只读文件中接下来k个字节作为一个字符返回                    |
| `fp.readline()`      | 从文件中读取一行内容，并以此作为一个字符串返回               |
| `fp.readlines()`     | 将文件中的每行内容作为一个字符串存入列表中，并返回该列表     |
| `fp.seek(k)`         | 将当前位置定位到文件的第k个字节                              |
| `for line in fp`     | 遍历文件中每一行                                             |
| `fp.tell()`          | 返回当前位置偏离开始处的字节数                               |
| `fp.write(string)`   | 在只写文件的当前位置将`string`的内容写入                     |
| `fp.writelines(seq)` | 在只写文件的当前位置写入给定序列的每个字符串。除了那些嵌入到字符串中的换行符，这个命令不插入换行符。 |

在磁盘上读写文件的功能都是由操作系统提供的，现代操作系统不允许普通的程序直接操作磁盘，所以，读写文件就是请求操作系统打开一个文件对象，然后，通过操作系统提供的接口从这个文件对象中读取数据，或者把数据写入这个文件对象。

$\text{file-like object}$：像`open()`函数返回的这种有个`read()`方法的对象，在Python中统称为file-like Object。除了`file`外，还可以是内存的字节流，网络流，自定义流等等。$\text{file-like object}$不要求从特定类继承，只要写个`read()`方法就行。`StringIO`就是在内存中创建的$\text{file-like object}$，常用作临时缓冲。

`StringIO`顾名思义就是在内存中读写`str`。`BytesIO`实现了在内存中读写`bytes`，我们创建一个`BytesIO`，然后写入一些`bytes`。

`dumps()`方法返回一个`str`，内容就是标准的`JSON`。类似的，`dump()`方法可以直接把`JSON`写入一个`file-like Object`。要把`JSON`反序列化为Python对象，用`loads()`或者对应的`load()`方法，前者把`JSON`的字符串反序列化，后者从`file-like Object`中读取字符串并反序列化

###### 异步io

在一个线程中，CPU执行代码的速度极快，然而，一旦遇到IO操作，就需要等待IO操作完成，才能继续进行下一步操作。这种情况称为同步IO。在IO操作的过程中，当前线程被挂起，而其他需要CPU执行的代码就无法被当前线程执行了。因为一个IO操作就阻塞了当前线程，导致其他代码无法执行，所以我们必须使用多线程或者多进程来并发执行代码，为多个用户服务。每个用户都会分配一个线程，如果遇到IO导致线程被挂起，其他用户的线程不受影响。多线程和多进程的模型虽然解决了并发问题，但是系统不能无上限地增加线程。由于系统切换线程的开销也很大，所以，一旦线程数量过多，CPU的时间就花在线程切换上了，真正运行代码的时间就少了，结果导致性能严重下降。由于我们要解决的问题是CPU高速执行能力和IO设备的龟速严重不匹配，多线程和多进程只是解决这一问题的一种方法。另一种解决IO问题的方法是异步IO。当代码需要执行一个耗时的IO操作时，它只发出IO指令，并不等待IO结果，然后就去执行其他代码了。一段时间后，当IO返回结果时，再通知CPU进行处理。

```python
do_some_code()
f = open('/path/to/file', 'r')
r = f.read() # <== 线程停在此处等待IO操作结果
# IO操作完成后线程才能继续执行:
do_some_code(r)
#异步IO模型需要一个消息循环，在消息循环中，主线程不断地重复“读取消息-处理消息”这一过程
loop = get_event_loop()
while True:
    event = loop.get_event()
    process_event(event)
```

消息模型其实早在应用在桌面应用程序中了。一个GUI程序的主线程就负责不停地读取消息并处理消息。所有的键盘、鼠标等消息都被发送到GUI程序的消息队列中，然后由GUI程序的主线程处理。由于GUI线程处理键盘、鼠标等消息的速度非常快，所以用户感觉不到延迟。某些时候，GUI线程在一个消息处理的过程中遇到问题导致一次消息处理时间过长，此时，用户会感觉到整个GUI程序停止响应了。这种情况说明在消息模型中，处理一个消息必须非常迅速，否则，主线程将无法及时处理消息队列中的其他消息，导致程序看上去停止响应。

消息模型是如何解决同步IO必须等待IO操作这一问题的呢？当遇到IO操作时，代码只负责发出IO请求，不等待IO结果，然后直接结束本轮消息处理，进入下一轮消息处理过程。当IO操作完成后，将收到一条“IO完成”的消息，处理该消息时就可以直接获取IO操作结果。在“发出IO请求”到收到“IO完成”的这段时间里，同步IO模型下，主线程只能挂起，但异步IO模型下，主线程并没有休息，而是在消息循环中继续处理其他消息。这样，在异步IO模型下，一个线程就可以同时处理多个IO请求，并且没有切换线程的操作。对于大多数IO密集型的应用程序，使用异步IO将大大提升系统的多任务处理能力。

###### 协程

子程序，或者称为函数，在所有语言中都是层级调用，比如A调用B，B在执行过程中又调用了C，C执行完毕返回，B执行完毕返回，最后是A执行完毕。所以子程序调用是通过栈实现的，一个线程就是执行一个子程序。子程序调用总是一个入口，一次返回，调用顺序是明确的。

协程的调用和子程序不同。协程看上去也是子程序，但执行过程中，在子程序内部可中断，然后转而执行别的子程序，在适当的时候再返回来接着执行。

##### 文件和目录访问

###### pathlib

该模块提供表示文件系统路径的类，其语义适用于不同的操作系统。路径类被分为提供纯计算操作而没有 I/O 的 纯路径，以及从纯路径继承而来但提供 I/O 操作的 具体路径。

![](./picture/2/42.png)

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

###### os.path

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

###### os

获取当前工作目录：`os.getcwd`
创建单个目录：`os.mkdir`
创建嵌套目录结构：`os.makedirs`  `os.makedirs('one/two')`
删除目录：`os.rmdir`：不能删除非空目录
删除多个目录：`os.rmdirs`：递归删除树结构中的目录。