- **什么是数据漂移？**


通常是指ods表的同一个业务日期数据中包含了前一天或后一天凌晨附近的数据或者丢失当天变更的数据，这种现象就叫做漂移，且在大部分公司中都会遇到的场景。

**如何解决数据漂移问题？**

通常有两种解决方案：

1. 多获取后一天的数据，保障数据只多不少

2. 通过多个时间戳字段来限制时间获取相对准确的数据

第一种方案比较暴力，这里不做过多解释，主要来讲解一下第二种解决方案。（首先这种解决方案在大数据之路这本书有体现）

**以下内容为该书的描述：**

通常，时间戳字段分为四类：

1. 数据库表中用来标识**数据记录更新时间**的时间戳字段（假设这类字段叫 modified time ）

2. 数据库**日志中**用来标识**数据记录更新时间**的时间戳字段·（假设这类宇段叫 log_time）

3. 数据库表中用来记录**具体业务过程发生时间**的时间戳字段 （假设这类字段叫 proc_time）

4. 标识数据记录**被抽取到时间**的时间戳字段（假设这类字段extract time）

理论上这几个时间应该是一致的，但往往会出现差异，造成的原因可能为：

1. 数据抽取需要一定的时间，extract_time往往晚于前三个时间

2. 业务系统手动改动数据并未更新modfied_time

3. 网络或系统压力问题，log_time或modified_time晚于proc_time

通常都是根据以上的某几个字段来切分ODS表，这就产生了数据漂移。具体场景如下：

1. 根据extract_time进行同步

2. 根据modified_time进行限制同步， 在实际生产中这种情况最常见，但是往往会发生不更新 modified time 而导致的数据遗漏，或者凌晨时间产生的数据记录漂移到后天 。由于网络或者系统压力问题， log_time 会晚proc_time ，从而导致凌晨时间产生的数据记录漂移到后一天。

3. 根据proc_time来限制，会违背ods和业务库保持一致的原则，因为仅仅根据proc_time来限制，会遗漏很多其他过程的变化

那么该书籍中提到的**第二种解决方案**：

1. 首先通过log_time多同步前一天最后15分钟和后一天凌晨开始15分钟的数据，然后用modified_time过滤非当天的数据，这样确保数据不会因为系统问题被遗漏

2. 然后根据log_time获取后一天15分钟的数据，基于这部分数据，按照主键根据log_time做升序排序，那么第一条数据也就是最接近当天记录变化的

3. 最后将前两步的数据做全外连接，通过限制业务时间proc_time来获取想要的数据

数据库在通过连接两张或多张表来返回记录时，都会生成一张中间的临时表

以 LEFT JOIN 为例：在使用 LEFT JOIN 时，ON 和 WHERE 过滤条件的区别如下：

**on** 条件是在生成临时表时使用的条件，它不管 **on** 中的条件是否为真，都会返回左边表中的记录

**where** 条件是在临时表生成好后，再对临时表进行过滤的条件。这时已经没有 **left join** 的含义（必须返回左边表的记录）了，条件不为真的就全部过滤掉。

> 缓慢变化维(Slowly Changing Dimensions) 指的是维度的属性并不是静态的，它会随着时间的流失发生缓慢的变化。比如像我们的工作经历就是属于缓慢变化的。一般针对这种变化信息处理有10种处理方式。
> 
> 1. 保留原值
>    
>    通常这种方式比较关注原始数据，比如原始的信用卡积分或者日期维度；需要对原始数据进行分析的场景下使用
> 
> 2. 重写覆盖
>    
>    即修改属性值为最新值，即只关心最新的变化，需要注意的是如果涉及到olap，可能会进行重复计算
> 
> 3. 增加新行
>    
>    通过追加新行的方式，需要注意的是和事实表的关联更新，即维度主键不能使用自然键或持久键，否则会出现数据发散。通常采用该种方式还需要增加几个附加列，如该行的有效时间和截止时间，以及当前行标示，这里可以通过拉链表来借助理解。
> 
> 4. 增加新属性列
>    
>    基于维表来增加新的属性列来保留历史属性值。一般不常用。
> 
> 5. 增加微型维度
>    
>    当维表中的部分属性出现快速变化的时候，可以使用该种方式，即将部分相对快速变化的属性从当前维表中划分出来，构建单独的微型维度表。
> 
> 6. 微型维度结合直接覆盖形成支架表
>    
>    在第5种方式的基础上再结合直接覆盖的方式。即建立微型维表后，微型维度的主键不仅作为事实表的外键，而且也是主维度的外键。当微型维度表的属性值发生变化的时候，直接进行覆盖。
> 
> 7. 同时增加行列，重写新加入的维度列
>    
>    这种方式的处理场景不常用，这里给出具体的样例配合理解。请注意截图标注的部分
>    
>    快照
> 
> 这种方式比较粗暴，即每天保留全量的快照数据，通过空间换时间
> 
> 9. 历史拉链
> 
> 拉链表的处理方式，即通过时间标示当前有效记录
> 
> 10. 双重外键结合直接覆盖和追加的方式
> 
> 基于追加新行的方式上，使用双键来区别历史/当前属性。通过代理键获取当前记录，使用自然键获取历史记录。也是通过样例配合理解

Hive窗口函数怎么设置窗口大小

这里主要说一下 current row ,preceding,following这几个关键词：

current row:表示当前行

UNBOUNDED PRECEDING：表示从组内的起点开始

num PRECEDING:表示从当前行的前num行开始

num FOLLOWING：表示截止到当前行的后num行

UNBOUNDED FOLLOWING:表示截止到组内的最后一行

这里只给出使用样例，根据调整上面的参数最后得到的结果会跟lag,lead函数是一样的

```hs
select 
 sum(cnt) over() as all_cnt,--所有行相加
 
 sum(cnt) over(partition by url) as url_cnt,--按url分组，组内数据相加
 
 sum(cnt) over(partition by url order by visit_time) as url_visit_asc_cnt,--按url分组，按照visit_time升序,组内数据累加
 
 sum(cnt) over(partition by url order by visit_time rows between UNBOUNDED PRECEDING and current row ) as url_cnt_1 ,--和sample3一样,由起点到当前行的聚合
 
sum(cnt) over(partition by url order by visit_time rows between 1 PRECEDING and current row) as url_cnt_2, --当前行和前面一行做聚合

sum(cnt) over(partition by url order by visit_time rows between 1 PRECEDING AND 1 FOLLOWING ) as url_cnt_3,--当前行和前边一行及后面一行

sum(cnt) over(partition by url order by visit_time rows between current row and UNBOUNDED FOLLOWING ) as url_cnt_4 --当前行及后面所有行

from wedw_dwd.log;

```

1. **order by**
   
   按照指定的key进行全局分组排序，且排序是全局有序的

2. **distributed by**
   
   distribute by 是控制map的输出在reducer端是如何划分的.hive会根据distribute by 后面指定的列，对应reducer的个数进行分发.默认采用hash算法.sort by 为每一个reduce产生一个排序文件.在有些情况下，你需要控制某个特定行应该到哪个reducer，这通常是为了进行后续的聚集操作。distribute by刚好可以做这件事。因此，distribute by经常和sort by配合使用.

3. **sort by**
   
   不是全局有序的，每个reduce端是有序的， 保证了局部有序 ；当只有一个reduce的时候，是可以实现全局有序的

4. **cluster by**
   
   是distirbuted by 和sort by 的结合，但是不能指定排序为asc或desc的规则，只能升序排列

首先数据同步根据读者使用情况来说明，而如何保证数据不丢失，笔者认为需要从以下3个方面来着手(其实就是从事前、事中、事后全方面考虑)

1.同步工具(事前)

目前主流的数据同步工具是sqoop、datax、canal、flume等。需要结合同步工具以及待同步的数据特性保证数据采集，例如使用sqoop来增量同步业务数据，这里需要保证业务数据中需要有更新时间，而且该更新时间是真的会进行更新(有时开发同学可能会忘记更新该时间的或者直接就是没有更新时间)。

2.平台稳定性(事中)

数据同步需要借助于调度系统，而且同步工具也有可能是直接集成到平台上，那么这个时候就需要保证调度和平台的稳定性。否则系统挂了还没有告警机制，那么就悲催了

3.监测机制(事后)

这里涉及到数据稽核了，即下游任务配置对应的稽核规则，当数据量波动超出阈值则需要发出告警，相关负责人就需要进行检查了。

以上是笔者个人的思路，描述可能不太全面或者准确，如果读者有更好的想法请及时联系笔者进行更改

hive支持的存储格式有TEXTFILE 、SEQUENCEFILE、ORC、PARQUET

| 压缩格式    | 算法      | 文件扩展名    | 是否可切分 | 对应的编码/解码器                                  |
| ------- | ------- | -------- | ----- | ------------------------------------------ |
| DEFLATE | DEFLATE | .deflate | 否     | org.apache.hadoop.io.compress.DefaultCodec |
| Gzip    | DEFLATE | .gz      | 否     | org.apache.hadoop.io.compress.GzipCodec    |
| bzip2   | bzip2   | .bz2     | 是     | org.apache.hadoop.io.compress.BZip2Codec   |
| LZO     | LZO     | .lzo     | 是     | com.hadoop.compression.lzo.LzopCodec       |
| Snappy  | Snappy  | .snappy  | 否     | org.apache.hadoop.io.compress.SnappyCodec  |

| 压缩算法  | 原始文件大小 | 压缩文件大小 | 压缩速度     | 解压速度     |
| ----- | ------ | ------ | -------- | -------- |
| gzip  | 8.3GB  | 1.8GB  | 17.5MB/s | 58MB/s   |
| bzip2 | 8.3GB  | 1.1GB  | 2.4MB/s  | 9.5MB/s  |
| LZO   | 8.3GB  | 2.9GB  | 49.3MB/s | 74.6MB/s |

| 参数                                               | 默认值                                                                                                                                                                       | 阶段        |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| io.compression.codecs （在core-site.xml中配置）        | org.apache.hadoop.io.compress.DefaultCodec, <br>org.apache.hadoop.io.compress.GzipCodec, org.apache.hadoop.io.compress.BZip2Codec, org.apache.hadoop.io.compress.Lz4Codec | 输入压缩      |
| mapreduce.map.output.compress                    | false                                                                                                                                                                     | mapper输出  |
| mapreduce.map.output.compress.codec              | org.apache.hadoop.io.compress.DefaultCodec                                                                                                                                | mapper输出  |
| mapreduce.output.fileoutputformat.compress       | false                                                                                                                                                                     | reducer输出 |
| mapreduce.output.fileoutputformat.compress.codec | org.apache.hadoop.io.compress. DefaultCodec                                                                                                                               | reducer输出 |
| mapreduce.output.fileoutputformat.compress.type  | RECORD                                                                                                                                                                    | reducer输出 |



