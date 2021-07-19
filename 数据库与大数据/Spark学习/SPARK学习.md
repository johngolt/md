对于提高并行度，对于RDD，需要从几个方面入手，1、配置num-executor。2、配置executor-cores。3、配置spark.default.parallelism。三者之间的关系一般为spark.default.parallelism=num-executors*executor-cores的2~3倍较为合适。对于Spark-sql，则设置spark.sql.shuffle.partitions、num-executor和executor-cores。



对Spark基本上就有认识了，分别是结构术语Shuffle、Patitions、MapReduce、Driver、Application Master、Container、Resource Manager、Node Manager等。API编程术语关键RDD、DataFrame，结构术语用于了解其运行原理，API术语用于使用过程中编写代码

相比于Spark RDD API，Spark SQL包含了对结构化数据和在其上运算的更多信息，Spark SQL使用这些信息进行了额外的优化，使对结构化数据的操作更加高效和方便。

DataFrame是一种以RDD为基础的分布式数据集，类似于传统数据库的二维表格，DataFrame带有Schema元信息，即DataFrame所表示的二维表数据集的每一列都带有名称和类型，但底层做了更多的优化。DataFrame可以从很多数据源构建，比如：已经存在的RDD、结构化文件、外部数据库、Hive表。

RDD可看作是分布式的对象的集合，Spark并不知道对象的详细模式信息，DataFrame可看作是分布式的Row对象的集合，其提供了由列组成的详细模式信息（就是列的名称和类型），使得Spark SQL可以进行某些形式的执行优化。DataFrame和普通的RDD的逻辑框架区别如下所示：

RDD是分布式的Java对象的集合。DataFrame是分布式的Row对象的集合。DataFrame除了提供了比RDD更丰富的算子以外，更重要的特点是提升执行效
率、减少数据读取以及执行计划的优化

![](../../picture/2/376.png)