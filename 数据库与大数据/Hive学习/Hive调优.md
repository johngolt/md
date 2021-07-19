### HIVE调优

#### `Hive on Spark`配置

##### `Executor`参数

假设我们使用的服务器单节点有32个CPU核心可供使用。考虑到系统基础服务和HDFS等组件的余量，一般会将YARN NodeManager的`yarn.nodemanager.resource.cpu-vcores`参数设为28，也就是YARN能够利用其中的28核，此时将`spark.executor.cores`设为4最合适，最多可以正好分配给7个Executor而不造成浪费。又假设`yarn.nodemanager.resource.cpu-vcores`为26，那么将`spark.executor.cores`设为5最合适，只会剩余1个核。
由于一个Executor需要一个YARN Container来运行，所以还需保证`spark.executor.cores`的值不能大于单个Container能申请到的最大核心数，即`yarn.scheduler.maximum-allocation-vcores`的值。

```
spark.executor.cores=n;
```

该参数表示每个Executor可利用的CPU核心数。其值不宜设定过大，因为Hive的底层以HDFS存储，而HDFS有时对高并发写入处理不太好，容易造成race condition。根据我们的实践，设定在3~6之间比较合理。

```
spark.executor.memory/spark.yarn.executor.memoryOverhead
```

这两个参数分别表示每个Executor可利用的堆内内存量和堆外内存量。堆内内存越大，Executor就能缓存更多的数据，在做诸如map join之类的操作时就会更快，但同时也会使得GC变得更麻烦。计算Executor总内存量的经验公式，如下：

`yarn.nodemanager.resource.memory-mb * (spark.executor.cores / yarn.nodemanager.resource.cpu-vcores)`

其实就是按核心数的比例分配。在计算出来的总内存量中，80%~85%划分给堆内内存，剩余的划分给堆外内存。假设集群中单节点有128G物理内存，`yarn.nodemanager.resource.memory-mb`（即单个NodeManager能够利用的主机内存量）设为120G，那么总内存量就是：120 * 1024 * (4 / 28) ≈ 17554MB。再按8:2比例划分的话，最终`spark.executor.memory`设为约13166MB，`spark.yarn.executor.memoryOverhead`设为约4389MB。

同时两个内存参数相加的总量也不能超过单个Container最多能申请到的内存量，即`yarn.scheduler.maximum-allocation-mb`。

```
spark.executor.instances
```

该参数表示执行查询时一共启动多少个Executor实例，这取决于每个节点的资源分配情况以及集群的节点数。若我们一共有10台32C/128G的节点，并按照上述配置（即每个节点承载7个Executor），那么理论上讲我们可以将`spark.executor.instances`设为70，以使集群资源最大化利用。但是实际上一般都会适当设小一些（推荐是理论值的一半左右），因为Driver也要占用资源，并且一个YARN集群往往还要承载除了Hive on Spark之外的其他业务。

```
spark.dynamicAllocation.enabled
```

上面所说的固定分配Executor数量的方式可能不太灵活，尤其是在Hive集群面向很多用户提供分析服务的情况下。所以更推荐将spark.dynamicAllocation.enabled参数设为true，以启用Executor动态分配。

##### `Driver`参数

```
spark.driver.cores
```

该参数表示每个Driver可利用的CPU核心数。绝大多数情况下设为1都够用。

```
spark.driver.memory/spark.driver.memoryOverhead
```

这两个参数分别表示每个Driver可利用的堆内内存量和堆外内存量。根据资源富余程度和作业的大小，一般是将总量控制在512MB~4GB之间，并且沿用Executor内存的“二八分配方式”。例如，spark.driver.memory可以设为约819MB，spark.driver.memoryOverhead设为约205MB，加起来正好1G。

##### Hive参数

绝大部分Hive参数的含义和调优方法都与on MR时相同，但仍有两个需要注意。

```
hive.auto.convert.join.noconditionaltask.size
```

我们知道，当Hive中做join操作的表有一方是小表时，如果hive.auto.convert.join和hive.auto.convert.join.noconditionaltask开关都为true（默认即如此），就会自动转换成比较高效的map-side join。而hive.auto.convert.join.noconditionaltask.size

但是Hive on MR下统计表的大小时，使用的是数据在磁盘上存储的近似大小，而Hive on Spark下则改用在内存中存储的近似大小。由于HDFS上的数据很有可能被压缩或序列化，使得大小减小，所以由MR迁移到Spark时要适当调高这个参数，以保证map join正常转换。一般会设为100~200MB左右，如果内存充裕，可以更大点。

```
hive.merge.sparkfiles
```

小文件是HDFS的天敌，所以Hive原生提供了合并小文件的选项，在on MR时是hive.merge.mapredfiles，但是on Spark时会改成hive.merge.sparkfiles，注意要把这个参数设为true。至于小文件合并的阈值参数，即hive.merge.smallfiles.avgsize与hive.merge.size.per.task都没有变化

```
spark.default.parallelism
```

该参数用于设置每个stage的默认task数量。这个参数极为重要，如果不设置可能会直接影响你的Spark作业性能。

参数调优建议：Spark作业的默认task数量为500~1000个较为合适。如果此参数不设置，那么Spark自己根据底层HDFS的block数量来设置task的数量，默认是一个HDFS block对应一个task。通常来说，Spark默认设置的数量是偏少的（比如就几十个task），如果task数量偏少的话，就会导致你前面设置好的Executor的参数都前功尽弃。建议的设置原则是，设置该参数为num-executors * executor-cores的2~3倍较为合适，比如Executor的总CPU core数量为300个，那么设置1000个task是可以的，此时可以充分地利用Spark集群的资源。

spark.storage.memoryFraction

参数说明：该参数用于设置RDD持久化数据在Executor内存中能占的比例，默认是0.6。也就是说，默认Executor 60%的内存，可以用来保存持久化的RDD数据。根据你选择的不同的持久化策略，如果内存不够时，可能数据就不会持久化，或者数据会写入磁盘。

参数调优建议：如果Spark作业中，有较多的RDD持久化操作，该参数的值可以适当提高一些，保证持久化的数据能够容纳在内存中。避免内存不够缓存所有的数据，导致数据只能写入磁盘中，降低了性能。但是如果Spark作业中的shuffle类操作比较多，而持久化操作比较少，那么这个参数的值适当降低一些比较合适。此外，如果发现作业由于频繁的gc导致运行缓慢（通过spark web ui可以观察到作业的gc耗时），意味着task执行用户代码的内存不够用，那么同样建议调低这个参数的值。

spark.shuffle.memoryFraction

参数说明：该参数用于设置shuffle过程中一个task拉取到上个stage的task的输出后，进行聚合操作时能够使用的Executor内存的比例，默认是0.2。也就是说，Executor默认只有20%的内存用来进行该操作。shuffle操作在进行聚合时，如果发现使用的内存超出了这个20%的限制，那么多余的数据就会溢写到磁盘文件中去，此时就会极大地降低性能。

参数调优建议：如果Spark作业中的RDD持久化操作较少，shuffle操作较多时，建议降低持久化操作的内存占比，提高shuffle操作的内存占比比例，避免shuffle过程中数据过多时内存不够用，必须溢写到磁盘上，降低了性能。此外，如果发现作业由于频繁的gc导致运行缓慢，意味着task执行用户代码的内存不够用，那么同样建议调低这个参数的值。

#### 配置相关的优化

##### 本地模式设置

对于数据量比较小的任务，避免启动查询触发任务造成时间过多浪费（远大于job执行时间），通过HIVE本地模式在单台机器上处理所有任务来缩短执行时间

```
--开启本地模式，hive会尝试使用本地模式执行其他的操作，可以避免触发一些MapReduce任务
set hive.exec.mode.local.auto = true;

--开启本地模式后，当输入文件大小小于此阈值时可以自动在本地模式运行，默认是 128兆。
hive.exec.mode.local.auto.inputbytes.max

-- 开启本地模式后，Hive Tasks小于此阈值时，可以自动在本地模式运行。
hive.exec.mode.local.auto.tasks.max
hive.mapred.local.mem//本地模式启动的JVM内存大小
```

##### 动态分区设置

```SPARQL
-- 开启或关闭动态分区
hive.exec.dynamic.partition=false;

-- 设置为nonstrict模式，让所有分区都动态配置，否则至少需要指定一个分区值
hive.exec.dynamic.partition.mode=strict;

-- 能被mapper或reducer创建的最大动态分区数，超出而报错
hive.exec.max.dynamic.partitions.pernode=100;

-- 一条带有动态分区SQL语句所能创建的最大动态分区总数，超过则报错
hive.exec.max.dynamic.partitions=1000;

-- 全局能被创建文件数目的最大值，通过Hadoop计数器跟踪，若超过则报错
hive.exec.max.created.files=100000;
-- 当有空分区生成时，是否抛出异常。一般不需要设置。
hive.error.on.empty.partition=false;
```

##### 并行相关设置

```bash
set hive.exec.parallel=true;    //开启任务并行执行
set hive.exec.parallel.thread.number=8; //最大线程数为8 

set hive.execution.engine=tez; //开启tez执行引擎
```

##### Fetch抓取

在Hive中，有些简单任务既可以转化为MR任务，也可以Fetch抓取，即直接读取table对应的存储目录下的文件得到结果，具体的行为取决于Hive的`hive.fetch.task.conversion`参数。

- 当设置为none时，所有任务转化为MR任务；
- 当设置为minimal时，全局查找`SELECT *`、在分区列上的`FILTER`(where...)、`LIMIT`才使用Fetch，其他为MR任务；
- 当设置为more时，不限定列，简单的查找`SELECT`、`FILTER`、`LIMIT`都使用Fetch，其他为MR任务。

##### `Map`端优化

`Map`端读数据时，由于读入数据文件大小分布不均匀，因此导致有些 Map Instance 读取并且处理的数据特别多，而有些 Map Instance 处理的数据特别少，造成 Map 端长尾

有中间层可用就用中间层，如果没有则看是否能分段跑。比如要取3个月的数据，则可以分别写三段sql，每段取一个月的数据。

###### 行列过滤

谓词下推指：将过滤表达式尽可能移动至靠近数据源的位置，以使真正执行时能直接跳过无关的数据。减少不必要的数据输入

列处理：在select中，只拿需要的列，如果有分区，尽量使用分区字段查询（分区过滤），避免使用select *全表扫描；行处理：两表连接时，对一个表的数据先where过滤，再join（如果先join再过滤，过滤的数据量会很大），即嵌套子查询

    select key from tablename where 分区字段 = '~'

###### 小文件合并

```
hive.merg.mapfiles=true;//合并map输出，用来控制是否merge MAP-ONLY型Job的文件输出
hive.merge.mapredfiles=false;//合并reduce输出，用来控制merge MAP-REDUCE型Job的文件输出
hive.merge.size.per.task=256*1000*1000：合并文件的大小
hive.mergejob.maponly=true：如果支持CombineHiveInputFormat则生成只有Map的任务执行merge。这个参数是用来控制是以MAP-ONLY的形式来进行merge（这里有个前提条件就是需要Hadoop支持CombineInputFormat，0.20之前的即使设置了这个参数true也不会生效）或者以MAP-REDUCE的形式来进行merge.
hive.merge.smallfiles.avgsize=16000000：文件的平均大小小于该值时，会启动一个MR任务执行merge。
```

###### `Map`优化

```
-- 每个map最小输入大小
set mapred.min.split.size = 30000000;

-- 每个map最大输入大小
set mapred.max.split.size = 100000000;
set mapred.min.split.size.per.node=100000000; 
set mapred.min.split.size.per.rack=100000000; 
--执行map前进行小文件合并
set hive.input.format = org.apache.hadoop.hive.ql.io.CombineHiveInputFormat; 
// 前面三个参数确定合并文件块的大小，大于文件块大小128m的，按照128m来分隔，小于128m,大于100m的，按照100m来分隔，把那些小于100m的（包括小文件和分隔大文件剩下的）， 进行合并
```

###### 大文件导致数据倾斜

当对文件使用GZIP压缩等不支持文件分割操作的压缩方式，在日后有作业涉及读取压缩后的文件时，该压缩文件只会被一个任务所读取。如果该压缩文件很大，则处理该文件的Map需要花费的时间会远多于读取普通文件的Map时间，该Map任务会成为作业运行的瓶颈。这种情况也就是Map读取文件的数据倾斜。只能将使用GZIP压缩等不支持文件分割的文件转为bzip和zip等支持文件分割的压缩方式。

##### `join`数据倾斜

###### `Map join`

小表与大表的连接，这种情况下Hive提供了mapjoin功能，通过将连接操作全部在Map任务中完成：没有Reduce任务，避免产生数据倾斜；没有Map、Reduce任务中间的shuffle操作，减少网络传输。

小表与大表Join时容易发生数据倾斜，表现为小表的数据量比较少但key却比较集中，导致分发到某一个或几个reduce上的数据比其他reduce多很多，造成数据倾斜。

map端join适用于当一张表很小(可以存在内存中)的情况，即可以将小表加载至内存。

```text
SET hive.auto.convert.join=true; --  默认true
SET hive.mapjoin.smalltable.filesize=600000000; -- 默认 25m
SET hive.auto.convert.join.noconditionaltask=true; -- 默认true，所以不需要指定map join hint
SET hive.auto.convert.join.noconditionaltask.size=10000000; -- 控制加载到内存的表的大小
```

一旦开启map端join配置，Hive会自动检查小表是否大于`hive.mapjoin.smalltable.filesize`配置的大小，如果大于则转为普通的join，如果小于则转为map端join。

![](../../picture/2/371.png)

首先，Task A(客户端本地执行的task)负责读取小表a，并将其转成一个HashTable的数据结构，写入到本地文件，之后将其加载至分布式缓存。然后，Task B任务会启动map任务读取大表b，在Map阶段，根据每条记录与分布式缓存中的a表对应的hashtable关联，并输出结果。注意：map端join没有reduce任务，所以map直接输出结果，即有多少个map任务就会产生多少个结果文件。

###### `Null`值或热点`key`处理

大表与大表Join时，当其中一张表的NULL值（或其他值）比较多时，容易导致这些相同值在reduce阶段集中在某一个或几个reduce上，发生数据倾斜问题。

`Null`值过滤：将NULL值提取出来最后合并，这一部分只有map操作；非NULL值的数据分散到不同reduce上，不会出现某个reduce任务数据加工时间过长的情况，整体效率提升明显。

```
select b.* from (select * from B where id is not null) b left join C c on b.id = c.id;
```

`Null`值转换：有时虽然某个key对应的null很多，但null并不是异常数据，不能过滤掉，必须包含在join的结果中，这样就可以考虑把表中key为null的字段赋一个随机值，使得数据随机均匀分到不同的reducer上。适合无效字段（id=-99，null等）产生的数据倾斜问题。因为空值不参与关联，即使分到不同的reduce上，也不影响最终的结果。

```
首先设置reduce个数
set mapreduce.job.reduces = 5;
 
然后join两张表，随机设置null值
insert overwrite table A
select b.* from B b full join C c on
case when b.id is null then concat ('hive',rand()) else b.id end = c.id;
```

###### `skew join`

在执行JOIN的过程中，会将一个表中的大key（也就是倾斜的那部分数据，判断是否倾斜由配置项hive.skewjoin.key指定，默认是100000）输出到一个对应的目录中，同时该key在其他表中的数据输出到其他的目录中（每个表一个目录）。整个目录结构类似下面这样：

T1表中的大key存储在目录 dir-T1-bigkeys中，这些key在T2表的数据存储在dir-T2-keys中，这些key在T3表的数据存储在dir-T3-keys中；T2表中的大key在T1中的数据存储在 dir-T1-keys，T2表中的大key存储在目录dir-T2-bigkeys中，这些key在T3表的数据存储在dir-T3-keys中，以此类推；T3表中的大key在T1中的数据存储在 dir-T1-keys，这些key在T2表的数据存储在dir-T2-keys中，T3表中的大key存储在目录dir-T3-bigkeys中，以此类推。对于每个表，都会单独启动一个mapjoin作业处理，输入的数据，就是该表的大key的目录和其他表中这些key对应的目录，对于上面情况基本就是会启动三个map join作业

```
set hive.skewjoin.key=1000000; --这个是join的键对应的记录条数超过这个值则会进行分拆,值根据具体数据量设置
set hive.optimize.skewjoin=true;–如果是join 过程出现倾斜 应该设置为true
set hive.skewjoin.mapjoin.map.tasks=10000;//用于处理skew join的map join 的最大数量
set hive.skewjoin.mapjoin.min.split=33554432;
```

![](../../picture/2/373.png)

不超过hive.skewjoin.key的key，走正常join流程；超过的hive.skewjoin.key的key,先写到hdfs上，然后再启动一个join，执行map join

###### 不同数据类型

对于两个表join，表a中需要join的字段key为int，表b中key字段既有string类型也有int类型。当按照key进行两个表的join操作时，默认的Hash操作会按int型的id来进行分配，这样所有的string类型都被分配成同一个id，结果就是所有的string类型的字段进入到一个reduce中，引发数据倾斜。

如果key字段既有string类型也有int类型，默认的hash就都会按int类型来分配，那我们直接把int类型都转为string就好了，这样key字段都为string，hash时就按照string类型分配了：

```
SELECT *
FROM users a
 LEFT JOIN logs b ON a.usr_id = CAST(b.user_id AS string);
```

##### `group by`优化

Hive做group by查询，当遇到group by字段的某些值特别多的时候，会将相同值拉到同一个reduce任务进行聚合，也容易发生数据倾斜。

###### `Map`端聚合

```
set hive.map.aggr =true//通过设置属性hive.map.aggr值为true来提高聚合的性能，这个设置会触发在map阶段进行的‘顶级’聚合过程。默认为true
set hive.groupby.mapaggr.checkinterval=100000//再Map端进行聚合操作的条目数
hive.map.aggr.hash.min.reduction=0.5(默认)
解释：预先取100000条数据聚合,如果聚合后的条数小于100000*0.5，则不再聚合
```

Map端进行预聚合，减少shuffle数据量，类似于MR中的Combiner。

![](../../picture/2/374.png)

![](../../picture/2/375.png)

###### 负载均衡

```
set hive.groupby.skewindata=True;
```

当为`true`后，会生成两个MR Job，启两个任务。job1将group by的key，相同的key可能随机分发到不同的Reduce中，然后Reduce依据key对数据进行聚合，此时每一个Reduce中每个数据的key不一定相同，但是经过这一步聚合后，大大减少了数据量。job2是真正意义上MR的底层实现，将相同的key分配到同一个reduce中，进行key的聚合操作。第一步job1实现负载均衡，第二步job2实现聚合需求。

如果为`false`，也就是默认情况下，只会进行job2操作，进行一次MapReduce。

##### `Reduce`数据倾斜

reducer的数量并不是越多越好，我们知道有多少个reducer就会生成多少个文件，小文件过多在hdfs中就会占用大量的空间，造成资源的浪费。如果reducer数量过小，导致某个reducer处理大量的数据（数据倾斜就会出现这样的现象），没有利用hadoop的分而治之功能，甚至会产生OOM内存溢出的错误。使用多少个reducer处理数据和业务场景相关，不同的业务场景处理的办法不同。

```
hive.exec.reducers.tasks=3； 设置job任务的初始reducer个数hive的默认reducer个数是3
–每个reduce处理的最大数据量
set hive.exec.reducers.bytes.per.reducer=120000000;//默认值是1G，也就是说默认每个reducer默认处理最大1G数据量，我们可以通过改变这个配置数据量来改变reducer的个数。
hive.exec.reducers.max= ; 设定job任务最大启用reducer的个数，在集群中为了控制资源利用情况。建议设置值=（集群总reduce槽位个数*1.5）/（执行中查询的平均个数）

Reduce个数 N = min(参数2，总输入数据量/参数1)
```

其中set mapreduce.job.reduces=number的方式优先级最高，set hive.exec.reducers.max=number优先级次之，set hive.exec.reducers.bytes.per.reducer=number 优先级最低。

##### 中间数据压缩

对中间数据进行压缩可以减少job中map和reduce task间的数据传输量。但是数据压缩会增加cup的开销，选择一个低CPU开销的编/解码器要比选择一个压缩率高的编/解码器要重要得多。

```
set hive.exec.compress.intermediate=true;
set hive.intermediate.compression.codec=org.apache.hadoop.io.compress.SnappyCodec;
set hive.intermediate.compression.type=BLOCK;
hive.exec.compress.output=false; 最终输出结果压缩，默认为false
```

可以使用的压缩编解码器

```text
org.apache.hadoop.io.compress.DefaultCodec
org.apache.hadoop.io.compress.GzipCodec
org.apache.hadoop.io.compress.BZip2Codec
com.hadoop.compression.lzo.LzopCodec
org.apache.hadoop.io.compress.Lz4Codec
org.apache.hadoop.io.compress.SnappyCodec
```

##### 其他配置优化

```
set hive.optimize.bucketmapJOIN=true; //开启为true时才支持一张表的一个分桶数据与另一张表每个分桶进行匹配连接，当然前提条件是表中的数据必须按照ON语句的键进行分桶才行，而且其中一张表的分桶个数必须是另一张表的若干倍。

Set hive.mapred.mode=strict; 开启严格模式，在严格模式下hive要求（1.order by语句查询时必须加有limit；2.分区表查询必须加where分区查询；3.限制笛卡尔积的查询）设置为nonstrict为非严格模式
```

##### 具体场景优化

减少在select时用case when，实测发现case when的执行效率很低，当数据量太大的时候甚至会跑不出数

多用临时表，当需要建的表其逻辑非常复杂时，需要考虑用临时表的方式把中间逻辑分布执行，一来方便阅读、修改和维护，二来减少硬盘的开销

###### 用多次`group by`代替`count distinct`

```
select ds, count(distinct userid), count(order_id)
from(
     select ds, userid, order_id
     from table_a --- 子订单表，每订单含多个订单项
     group by ds, userid, order_id
)
group  by ds;
```

###### 全排序

```
select * from (select * from table distribute by time sort by time desc limit 50 ) t order by time desc limit 50;
```

###### `EXISTS/IN`子查询

Hive不支持 IN/EXISTS 子查询，左半连接是Hive对于 IN/EXISTS 子查询的一种更高效的实现。左半连接只传递右表用于比较的key，因此最后的结果只会有左表的数据，右表只能在on子句中设置过滤条件，在where等子句中不能过滤，遇到右表重复记录，左表会跳过，而内连接会一直遍历，因此在右表有重复记录时，左半连接仅生成一条记录，与IN相同

```
SELECT A.value1, A.value2
FROM A LEFT SEMI JOIN B ON A.value1=B.value1;
```

###### 多次INSERT单次扫描表

默认情况下，Hive会执行多次表扫描。因此，如果要在某张hive表中执行多个操作，建议使用一次扫描并使用该扫描来执行多个操作。

```
--multi insert 语句
FROM src
INSERT OVERWRITE TABLE dest1 SELECT src.* WHERE src.key < 100
INSERT OVERWRITE TABLE dest2 SELECT src.key, src.value WHERE src.key >= 100 and src.key < 200
```

