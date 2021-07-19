## HIVE学习

### `Hive`架构

![](../../picture/2/370.png)

用户接口层，与Hive进行交互，包括三种方式：`CLI`，即命令行接口，以交互的形式与Hive交互；`JDBC/ODBC`，用户使用不同的编程语言通过Thrift Server连接到Hive服务器；`Web UI`，浏览器访问Hive

Driver组件，Driver组件完成对`HQL`语句的语法分析、编译、优化，转化为MR任务并运行，具体地有解释器、编译器、优化器、执行器四个部分组成

`MetaStore`元数据存储组件，包括`MetaStore`服务和元数据的存储（即Hive中数据的描述，如表的名字、属性、位置等）。元数据默认存放在自带的`Derby`数据库中，但“不适合多用户操作，并且数据存储目录不固定”，不方便管理，因此一般会存放在自己创建的`Mysql`数据库中`MetaStore`服务即利用元数据与`Hive`进行交互的服务组件，`MetaStore`服务可以分离出来（即和Hive运行在不同的进程），如放到防火墙之后，可提高安全性和Hive效率

### `Hive`存储模型

![](../../picture/hive数据存储模型.jpg)

##### 内部表

内部表数据由Hive自身管理；内部表数据存储的位置是`hive.metastore.warehouse.dir`（默认：`/user/hive/warehouse`）； 删除内部表会直接删除元数据及存储数据； 对内部表的修改会将修改直接同步给元数据。如果所有处理都由Hive完成，建议使用内部表。

##### 外部表

外部表数据由`HDFS`管理；外部表数据的存储位置由自己制定；删除外部表仅仅会删除元数据，`HDFS`上的文件并不会被删除；对外部表的表结构和分区进行修改，则需要修复`MSCK REPAIR TABLE table_name;`。如果要用Hive和其他工具来处理同一个数据集，建议使用外部表。

##### 分区

分区使用的是`表外`字段，需要指定字段类型。分区通过`partitioned by(partition_name string)`声明。分区划分粒度较粗。将数据按区域划分开，查询时不用扫描无关的数据，加快查询速度。每个分区是一个目录。分区数量不固定。分区下可再有分区或者桶。

###### 动态和静态分区

静态分区与动态分区的主要区别在于静态分区是手动指定，而动态分区是通过数据来进行判断。静态分区的列是在编译时期，通过用户传递来决定的，动态分区只有在 SQL 执行时才能决定。

##### 分桶

分桶逻辑是对分桶字段求哈希值，用哈希值与分桶的数量取余决定数据放到哪个桶里。每个桶是一个文件。建表时指定桶个数，桶内可排序。

分桶使用的是`表内`字段。分桶表通过关键字`clustered by(column_name) into … buckets`声明。分桶是更细粒度的划分；分桶优点在于用于数据取样时候能够起到优化加速的作用。

#### 数据模型相关

Hive查询没有如果没有指定reduce key，hive会生成随机数作为reduce key。输入记录也随机地被分发到不同reducer。为了保证reducer之间没有重复的sale_id记录，可以使用DISTRIBUTE BY关键字指定分发key为sale_id。

```
set mapred.reduce.tasks=2; 
Select sale_id, amount from t_order Distribute by sale_id Sort by sale_id, amount;
```

##### `MapReduce`

`JOIN`转化为`MR`任务

- Map：生成键值对，以`JOIN ON`条件中的列作为`Key`，以`JOIN`之后所关心的列作为`Value`，在`Value`中还会包含表的 Tag 信息，用于标明此`Value`对应于哪个表
- Shuffle：根据`Key`的值进行 Hash，按照Hash值将键值对发送至不同的Reducer中
- Reduce：`Reducer`通过 Tag 来识别不同的表中的数据，根据`Key`值进行`Join`操作

![join操作](../../picture/2/353.png)

###### `group by`转化为`MR`任务

- Map：生成键值对，以`GROUP BY`条件中的列作为`Key`，以聚集函数的结果作为`Value`
- Shuffle：根据`Key`的值进行 Hash，按照Hash值将键值对发送至不同的Reducer中
- Reduce：根据`SELECT`子句的列以及聚集函数进行Reduce

![groupby操作](../../picture/2/354.png)

###### `Distinct`转化为`MR`任务

相当于没有聚集函数的`GROUP BY`，操作相同，只是键值对中的`value`可为空。

###### `Map`

![](../../picture/2/368.png)

hive作业会根据input目录产生一个或者多个map任务。map任务的个数主要由如下因素决定：input文件总个数、input文件的大小、集群设置的文件块大小 (目前为128`M`，该参数不能自定义修改)。假设input目录下有1个文件 c , 其大小为680`M`, `hadoop`会将该文件 c 切分为6个块（1个40`M`的和5个128`M`大小的块），对应的map数为6。

假设input目录下有4个文件a , b , c, d , 它们的大小分别为5`M`, 10`M`, 128`M`, 140`M`，那么`hadoop`会将其切分为5个块（5个块的大小分别为5`M`, 10`M`, 128`M`, 128`M`, 12`M`) ，对应的Map数是，即如果文件大于块大小(128`M`)，会进行拆分，如果小于块大小，则把该文件当成一个块进行处理。

map数并非越多越好，如果一个任务包含很多小文件（远远小于所设置的块大小），那么每个小文件都会被被当做一个独立的块且对应一个map。在上面这种情况下，map任务启动和初始化的时间远远大于逻辑处理的时间，造成很大的资源浪费。

如果表a只有一个文件，大小为120`M`，但包含几千万的记录，如果用1个map去完成这个任务，肯定是比较耗时的，这种情况下，我们要考虑将这一个文件合理的拆分成多个，这样就可以用多个map任务去完成，具体实现样例如下:

```
set mapred.reduce.tasks=10; 
create table tmp3 as 
select * from a
distribute by rand(20); 
```

这样会将a表的记录，随机的分散到包含20个文件的tmp3表中，再用tmp3 代替上面sql中的a 表，则会用20个map任务去完成。每个map任务处理6M左右（数百万记录）的数据，效率会提高不少。

###### `Reduce`

![](../../picture/2/369.png)

reduce的启动同样会耗费时间与计算资源。 另外reduce的个数决定着输出文件个数，如何reduce处理之后生成了很多小文件，如果不经过处理就传递给下游的话，又会出现小文件过多的问题。
在设置reduce个数的时候同样需要考虑如下两个原则：即大数据量情况下配置合适的reduce数；单个reduce任务处理合适的数据量。`set mapred.reduce.tasks = 15;`如果在命令中明确配置reduce个数，那么hive就不会推测reduce个数，而是直接创建15个reduce任务。

### `HQL`

#### `Hive`数据类型

##### 基本数据类型

| 类型        | 范围                              |
| ----------- | --------------------------------- |
| `TINYINT`   | 1字节的有符号整数 -128~127        |
| `SMALINT`   | 2个字节的有符号整数，-32768~32767 |
| `INT`       | 4个字节的带符号整数               |
| `BIGINT`    | 8字节带符号整数                   |
| `BOOLEAN`   | 布尔型                            |
| `FLOAT`     | 4字节单精度浮点数                 |
| `DOUBLE`    | 8字节双精度浮点数                 |
| `DECIMAL`   | 任意精度的带符号小数              |
| `STRING`    | 字符串，变长                      |
| `CHAR`      | 固定长度字符串                    |
| `VARCHAR`   | 变长字符串                        |
| `TIMESTAMP` | 时间戳，纳秒精度                  |
| `DATE`      | 日期类型                          |
| `BINARY`    | 字节序列                          |

##### 集合数据类型

| 类型     | 语法示例   |
| -------- | ---------- |
| `STRUCT` | `struct()` |
| `MAP`    | `map()`    |
| `ARRAY`  | `array()`  |

Hive中存在隐式转换和cast函数转换两种方法对数据类型进行变更：

隐式类型转换的规则：任何整数类型都可以隐式地转换为一个范围更广的类型，如`TINYINT`可以转换成`INT`，`INT`可以转换成`BIGINT`；所有整数类型、FLOAT和STRING类型都可以隐式地转换成DOUBLE；`TINYINT`、`SMALLINT`、`INT`都可以转换为`FLOAT`；`BOOLEAN`类型不可以转换为任何其它的类型。

使用CAST操作显式进行数据类型转换：例如CAST('1' AS INT)将把字符串'1' 转换成整数1；如果强制类型转换失败，如执行CAST('X' AS INT)，表达式返回空值 NULL。需要注意的是，将浮点数转换成整数的推荐方式是使用round()或者floor()函数，而不是使用cast。