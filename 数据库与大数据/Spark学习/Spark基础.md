master和worker是物理节点，driver和executor是进程。搭建spark集群的时候我们就已经设置好了master节点和worker节点，一个集群有多个master节点和多个worker节点。master节点常驻master守护进程，负责管理worker节点，我们从master节点提交应用。worker节点常驻worker守护进程，与master节点通信，并且管理executor进程。一台机器可以同时作为master和worker节点

![](../../picture/2/372.png)

driver进程就是应用的main()函数并且构建sparkContext对象，当我们提交了应用之后，便会启动一个对应的driver进程，driver本身会根据我们设置的参数占有一定的资源（主要指cpu core和memory）。

driver可以运行在master上，也可以运行worker上（根据部署模式的不同）。driver首先会向集群管理者`standalone、yarn，mesos`申请spark应用所需的资源，也就是executor，然后集群管理者会根据spark应用所设置的参数在各个worker上分配一定数量的executor，每个executor都占用一定数量的cpu和memory。在申请到应用所需的资源以后，driver就开始调度和执行我们编写的应用代码了。driver进程会将我们编写的spark应用代码拆分成多个stage，每个stage执行一部分代码片段，并为每个stage创建一批tasks，然后将这些tasks分配到各个executor中执行。

executor进程宿主在worker节点上，一个worker可以有多个executor。每个executor持有一个线程池，每个线程可以执行一个task，executor执行完task以后将结果返回给driver，每个executor执行的task都属于同一个应用。此外executor还有一个功能就是为应用程序中要求缓存的 RDD 提供内存式存储，RDD 是直接缓存在executor进程内的，因此任务可以在运行时充分利用缓存数据加速运算。


Spark主要用于替代Hadoop中的MapReduce计算模型。存储依然可以使用HDFS，但是中间结果可以存放在内存中；调度可以使用Spark内置的，也可以使用更成熟的调度系统YARN等。
MapReduce框架采用非循环式的数据流模型，把中间结果写入到HDFS中，带来了大量的数据复制、磁盘IO和序列化开销。

RDD介绍的注释，我们来翻译下：
A list of partitions ：一组分片(Partition)/一个分区(Partition)列表，即数据集的基本组成单位。对于RDD来说，每个分片都会被一个计算任务处理，分片数决定并行度。用户可以在创建RDD时指定RDD的分片个数，如果没有指定，那么就会采用默认值。
A function for computing each split ：一个函数会被作用在每一个分区。Spark中RDD的计算是以分片为单位的，compute函数会被作用到每个分区上。
A list of dependencies on other RDDs ：一个RDD会依赖于其他多个RDD。RDD的每次转换都会生成一个新的RDD，所以RDD之间就会形成类似于流水线一样的前后依赖关系。在部分分区数据丢失时，Spark可以通过这个依赖关系重新计算丢失的分区数据，而不是对RDD的所有分区进行重新计算。(Spark的容错机制)
Optionally, a Partitioner for key-value RDDs (e.g. to say that the RDD is hash-partitioned)：可选项，对于KV类型的RDD会有一个Partitioner，即RDD的分区函数，默认为HashPartitioner。
RDD的算子分为两类:
Transformation转换操作:返回一个新的RDD
Action动作操作:返回值不是RDD(无返回值或返回其他的)

1、RDD不实际存储真正要计算的数据，而是记录了数据的位置在哪里，数据的转换关系(调用了什么方法，传入什么函数)。
2、RDD中的所有转换都是惰性求值/延迟执行的，也就是说并不会直接计算。只有当发生一个要求返回结果给Driver的Action动作时，这些转换才会真正运行。
窄依赖:父RDD的一个分区只会被子RDD的一个分区依赖；
宽依赖:父RDD的一个分区会被子RDD的多个分区依赖(涉及到shuffle)。
窄依赖：父RDD的一个分区只会被子RDD的一个分区依赖。即一对一或者多对一的关系，可理解为独生子女。 常见的窄依赖有：map、filter、union、mapPartitions、mapValues、join（父RDD是hash-partitioned）等。      

![](../../picture/spark依赖.jpg)

宽依赖：父RDD的一个分区会被子RDD的多个分区依赖(涉及到shuffle)。即一对多的关系，可理解为超生。常见的宽依赖有groupByKey、partitionBy、reduceByKey、join（父RDD不是hash-partitioned）等。

对于窄依赖：
窄依赖的多个分区可以并行计算；
窄依赖的一个分区的数据如果丢失只需要重新计算对应的分区的数据就可以了。
对于宽依赖：
划分Stage(阶段)的依据:对于宽依赖,必须等到上一阶段计算完成才能计算下一阶段。
DAG的边界
开始:通过SparkContext创建的RDD；
结束:触发Action，一旦触发Action就形成了一个完整的DAG
一个Spark程序可以有多个DAG(有几个Action，就有几个DAG，上图最后只有一个Action（图中未表现）,那么就是一个DAG)。
一个DAG可以有多个Stage(根据宽依赖/shuffle进行划分)。
同一个Stage可以有多个Task并行执行(task数=分区数，

Spark on Hive : Hive只作为存储角色，Spark负责sql解析优化，执行。
这里可以理解为Spark 通过Spark SQL 使用Hive 语句操作Hive表 ,底层运行的还是 Spark RDD。具体步骤如下：

通过SparkSQL，加载Hive的配置文件，获取到Hive的元数据信息；
获取到Hive的元数据信息之后可以拿到Hive表的数据；
通过SparkSQL来操作Hive表中的数据。
Hive on Spark：Hive既作为存储又负责sql的解析优化，Spark负责执行。
这里Hive的执行引擎变成了Spark，不再是MR，相较于Spark on Hive，这个实现较为麻烦，必须要重新编译spark并导入相关jar包。目前，大部分使用Spark on Hive。