①尽量避免在字段开头模糊查询，会导致数据库引擎放弃索引进行全表扫描 SELECT * FROM t WHERE username LIKE '%陈%'

②尽量避免使用 in 和 not in，会导致引擎走全表扫描,如果是连续数值，可以用 between 代替。如果是子查询，可以用 exists 代替。

```sql
-- 不走索引
select * from A where A.id in (select id from B);
-- 走索引
select * from A where exists (select * from B where B.id = A.id);
```

③尽量避免使用 or，会导致数据库引擎放弃索引进行全表扫描,可以用 union 代替 or。

⑤尽量避免在 where 条件中等号的左侧进行表达式、函数操作，会导致数据库引擎放弃索引进行全表扫描, 可以将表达式、函数操作移动到等号右侧。应尽量避免在where子句中对字段进行表达式操作，应尽量避免在where子句中对字段进行函数操作。 

④尽量避免进行 null 值的判断，会导致数据库引擎放弃索引进行全表扫描,可以给字段添加默认值 0，对 0 值进行判断。

③多表关联查询时，小表在前，大表在后。MySQL 采用从左往右，自上而下的顺序解析 where 子句。根据这个原理，应将过滤数据多的条件往前放，最快速度缩小结果集。

⑦查询条件不能用 <> 或者 !=，使用索引列作为条件进行查询时，需要避免使用<>或者!=等判断条件。应尽量避免在where子句中使用!=或<>操作符，MySQL只有对以下操作符才使用索引：<，<=，=，>，>=，BETWEEN，IN，以及某些时候的LIKE。

⑧where 条件仅包含复合索引非前置列
如下：复合（联合）索引包含 key_part1，key_part2，key_part3 三列，但 SQL 语句没有包含索引前置列"key_part1"，按照 MySQL 联合索引的最左匹配原则，不会走联合索引。

⑨隐式类型转换造成不使用索引
如下 SQL 语句由于索引对列类型为 varchar，但给定的值为数值，涉及隐式类型转换，造成不能正确走索引。
select col1 from table where col_varchar=123; 
⑩order by 条件要与 where 中条件一致，否则 order by 不会利用索引进行排序

```sql
-- 不走age索引
SELECT * FROM t order by age;
-- 走age索引
SELECT * FROM t where age > 0 order by age;
```

MySQL 通过创建并填充临时表的方式来执行 union 查询。除非确实要消除重复的行，否则建议使用 union all。
