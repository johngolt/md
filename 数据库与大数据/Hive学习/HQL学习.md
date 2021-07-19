### `HQL`

##### `DDL`

```
USE database_name;
--查询有哪些数据库
SHOW DATABASES

--查询有哪些表
SHOW TABLES IN database_name

--查询一个表有哪些分区
SHOW PARTITIONS table_name

--查询建表语句
SHOW CREATE TABLE ([db_name.]table_name|view_name);

--查询一个表有哪些字段
SHOW COLUMNS (FROM|IN) table_name [(FROM|IN) db_name];

--查询配置
show conf 'hive.exec.reducers.max';

--查询数据库、表、分区的描述信息
DESCRIBE DATABASE [EXTENDED] db_name;
DESCRIBE [EXTENDED|FORMATTED] table_name
DESCRIBE extended part_table partition (d='abc');
```

###### `CREATE`

```
CREATE [TEMPORARY] [EXTERNAL] TABLE [IF NOT EXISTS] [db_name.]table_name
  [(col_name data_type [column_constraint_specification] [COMMENT col_comment], ... [constraint_specification])]
  [COMMENT table_comment]
  [PARTITIONED BY (col_name data_type [COMMENT col_comment], ...)]
  [CLUSTERED BY (col_name, col_name, ...) [SORTED BY (col_name [ASC|DESC], ...)] INTO num_buckets BUCKETS]
  [SKEWED BY (col_name, col_name, ...)]
     ON ((col_value, col_value, ...), (col_value, col_value, ...), ...)
     [STORED AS DIRECTORIES]
  [
   [ROW FORMAT row_format] 
   [STORED AS file_format]
     | STORED BY 'storage.handler.class.name' [WITH SERDEPROPERTIES (...)]
  ]
  [LOCATION hdfs_path]
  [TBLPROPERTIES (property_name=property_value, ...)]
  [AS select_statement];
```

CLUSTERED BY 表示分桶表，按什么字段分区和排序。INTO 表示根据这个字段分多少个桶。SKEWED BY 表示指定某些列上有倾斜值，Hive 会记录下这些值，在查询的时候，会有更好的性能表现；STORED AS 表示以什么压缩格式来存储。LOCATION 是指定外部表的存储路径，MANAGEDLOCATION 是指定管理表的存储路径

create table as 语法，表示以目标的查询结果来创建表

create table like 语法，表示以 like 后面的表来创建表结构，不写数据进去

```
--创建视图
CREATE VIEW onion_referrers(url COMMENT 'URL of Referring page')
  COMMENT 'Referrers to The Onion website'
  AS
  SELECT DISTINCT referrer_url
  FROM page_view
  WHERE page_url='http://www.theonion.com';
```

###### `DROP`

```
DROP (DATABASE|SCHEMA) [IF EXISTS] database_name [RESTRICT|CASCADE];
```

如果数据库下有表，则不允许删除；如果要删除，后面加 CASCADE。RESTRICT 为默认值，默认不允许删除。

```
--清空表的所有数据，或者分区的所有数据
TRUNCATE [TABLE] table_name [PARTITION partition_spec];
--删除视图
DROP VIEW IF EXISTS onion_referrers;
```

###### `ALTER`

```
-- 重命名表
ALTER TABLE table_name RENAME TO new_table_name;
--修改表属性
--ALTER TABLE table_name SET TBLPROPERTIES table_properties;
alter table table_name set TBLPROPERTIES ('EXTERNAL'='TRUE') --内部表转内部表
```

```
--修改列名、列类型
ALTER TABLE table_name [PARTITION partition_spec] CHANGE [COLUMN] col_old_name col_new_name column_type
  [COMMENT col_comment] [FIRST|AFTER column_name] [CASCADE|RESTRICT];
--增加列
ALTER TABLE name ADD COLUMNS (col_spec[, col_spec ...])
alter table table_name add columns (column_new_name column_new_type [comment 'comment'])
ALTER TABLE name DROP [COLUMN] column_name
ALTER TABLE name CHANGE column_name new_name new_type
--更新列
ALTER TABLE name REPLACE COLUMNS (col_spec[, col_spec ...])
```

```
--增加分区
ALTER TABLE page_view ADD PARTITION (dt='2008-08-08', country='us') location '/path/to/us/part080808'
                          PARTITION (dt='2008-08-09', country='us') location '/path/to/us/part080809';
--删除分区
alter table pt_table drop if exists partition(dt='20201020');
--修改分区名
alter table table_name partition（dt='partition_old_name'） rename to partition(dt='partition_new_name')
--修改分区属性
alter table table_name partition column (dt partition_new_type)
--修改分区位置
alter table table_name partition (createtime='20190301') set location "new_location"

```

###### `INSERT`

```
-- insert overwrite
INSERT OVERWRITE TABLE tablename1 [PARTITION (partcol1=val1, partcol2=val2 ...) [IF NOT EXISTS]] 
select_statement1 FROM from_statement;

-- insert into
INSERT INTO TABLE tablename1 [PARTITION (partcol1=val1, partcol2=val2 ...)] 
select_statement1 FROM from_statement;
```



```
--将查询结果写入文件系统
INSERT OVERWRITE [LOCAL] DIRECTORY directory1
  [ROW FORMAT row_format] [STORED AS file_format]
  SELECT ... FROM ...


INSERT OVERWRITE LOCAL DIRECTORY '/tmp/reg_3' SELECT a.* FROM events a;
```

```
--Common Table Expression
--CTE 可以把一个临时的查询结果放到 with 语法中，供多个语法块使用
with q1 as ( select key from q2 where key = '5'),
q2 as ( select key from src where key = '5')
select * from (select key from q1) a;

FROM page_view_stg pvs
INSERT OVERWRITE TABLE page_view PARTITION(dt='2008-06-08', country)
       SELECT pvs.viewTime, pvs.userid, pvs.page_url, pvs.referrer_url, null, null, pvs.ip, pvs.cnt
```



#### 数据操作`DML`

##### 数据导入

使用LOAD DATA来存储大量记录。有两种方法用来加载数据：一种是从本地文件系统，第二种是从Hadoop文件系统。加载数据的语法如下：

```
LOAD DATA [LOCAL] INPATH 'filepath' [OVERWRITE] INTO TABLE tablename 
[PARTITION (partcol1=val1, partcol2=val2 ...)]
```

`LOCAL`是标识符指定本地路径。它是可选的。`inpath`:表示加载数据的路径。`OVERWRITE `是可选的，覆盖表中的数据，否则表示追加。`PARTITION`这是可选的

##### 查询

###### `SELECT`

```
SELECT [ALL | DISTINCT] select_expr, select_expr, ... 
FROM table_reference 
[WHERE where_condition] 
[GROUP BY col_list] 
[HAVING having_condition] 
[CLUSTER BY col_list | [DISTRIBUTE BY col_list] [SORT BY col_list]] 
[LIMIT number];
```

#### `HQL`函数



##### 日期相关函数

##### 窗口函数

窗口函数都是`最后一步执行`，而且仅位于 Order by 字句之前。窗口函数的计算“过程”如下：按窗口定义，将所有输入数据分区、再排序；对每一行数据，计算它的 Frame 范围；将 Frame 内的行集合输入窗口函数，计算结果填入当前行

```
window_function (expression) OVER (
   [ PARTITION BY part_list ]
   [ ORDER BY order_list ]
   [ { ROWS | RANGE } BETWEEN frame_start AND frame_end ] )
```

PARTITION BY 表示将数据先按 part_list 进行分区；ORDER BY 表示将各个分区内的数据按 order_list 进行排序。ROWS 选择前后几行，例如 ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING 表示往前 3 行到往后 3 行，一共 7 行数据（或小于 7 行，如果碰到了边界）；RANGE 选择数据范围，例如 RANGE BETWEEN 3 PRECEDING AND 3 FOLLOWING 表示所有值在`[cc−3,cc+3]` 这个范围内的行，cc 为当前行的值(range后面跟计算表达式，对order by后面的某个字段值进行计算，计算后的结果表示其真正的范围。)

1. CURRENT ROW：当前行

2. n PRECEDING：往前 n 行数据

3. n FOLLOWING：往后 n 行数据

4. UNBOUNDED：起点，UNBOUNDED PRECEDING 表示从前面的起点， UNBOUNDED FOLLOWING 表示到后面的终点

如果不指定 PARTITION BY，则不对数据进行分区；换句话说，所有数据看作同一个分区，如果不指定 ORDER BY，则不对各分区做排序，通常用于那些顺序无关的窗口函数

如果不指定 Frame 子句，则默认采用以下的 Frame 定义：若不指定 ORDER BY，默认使用分区内所有行 RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING；若指定了 ORDER BY，默认使用分区内第一行到当前值 RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW

窗口函数可以分为以下 3 类：

| 取值                  | 作用                                     |
| --------------------- | ---------------------------------------- |
| `FIRST_VALUE()`       | 取分组内排序后，截止到当前行，第一个值   |
| `LAST_VALUE()`        | 取分组内排序后，截止到当前行，最后一个值 |
| `LEAD(col,n,DEFAULT)` | 用于统计窗口内往下第n行值                |
| `LAG(col,n,DEFAULT)`  | 用于统计窗口内往上第n行值                |

排名函数不支持window子句，即不支持自定义窗口大小

| 排序           | 作用                                            |
| -------------- | ----------------------------------------------- |
| `RANK()`       | 有并列，相同名次空位(类似于1 1 3)               |
| `DENSE_RANK()` | 有并列，相同名次不空位（类似于1 1 2）           |
| `ROW_NUMBER()` | 没有并列，相同名次顺序排序                      |
| `NTILE(n)`     | 用于将分组数据按照顺序切分成n片，返回当前切片值 |

![](../../picture/窗口函数.jpg)
