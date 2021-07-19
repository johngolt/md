- order订单表，字段为：goods_id， amount ；  

- pv 浏览表，字段为：goods_id，uid；  

- goods按照总销售金额排序，分成top10,top10~top20,其他三组

求每组商品的浏览用户数（同组内同一用户只能算一次）

```sql
create table if not exists test.nil_goods_category as 
select goods_id
,case when nn<= 10 then 'top10'
      when nn<= 20 then 'top10~top20'
      else 'other' end as goods_group
from
(
    select goods_id
    ,row_number() over(partition by goods_id order by sale_sum desc) as nn
    from
    (
        select goods_id,sum(amount) as sale_sum
        from order 
        group by 1
    ) aa
) bb;
select b.goods_group,count(distinct a.uid) as num
from pv a 
left join test.nil_goods_category b 
on a.goods_id = b.goods_id
group by 1;
```

商品活动表 goods_event，g_id（有可能重复），t1（开始时间）,t2（结束时间）；给定时间段（t3,t4)，求在时间段内做活动的商品数

```sql
select count(distinct g_id) as event_goods_num
from goods_event
where (t1<=t4 and t1>=t3) 
or (t2>=t3 and t2<=t4)

select count(distinct g_id) as event_goods_num
from goods_event
where (t1<=t4 and t1>=t3) 
union all
```

商品活动流水表，表名为event，字段：goods_id， time；  求参加活动次数最多的商品的最近一次参加活动的时间

```sql
select a.goods_id,a.time
from event a 
inner join
(
    select goods_id,count(*)
    from event
    group by gooods_id
    order by count(*) desc
    limit 1
) b
on a.goods_id = b.goods_id
order by a.goods_id,a.time desc
```

用户登录的log数据，划定session，同一个用户一个小时之内的登录算一个session；生成session列

```sql
drop table if exists koo.nil_temp0222_a2;
create table if not exists koo.nil_temp0222_a2 as
select *
    ,row_number() over(partition by userid order by inserttime) as nn1
from 
(
    select a.*
    ,b.inserttime as inserttime_aftr
    ,datediff(b.inserttime,a.inserttime) as session_diff
  from
  (
    select userid,inserttime
      ,row_number() over(partition by userid order by inserttime asc) nn
    from koo.nil_temp0222 
    where userid = 1900000169
  ) a   
  left join 
  (
     select userid,inserttime
      ,row_number() over(partition by userid order by inserttime asc) nn
    from koo.nil_temp0222 
    where userid = 1900000169 
  ) b
  on a.userid =  b.userid and a.nn = b.nn-1
) aa
where session_diff >10 or nn = 1
order by userid,inserttime;

drop table if exists koo.nil_temp0222_a2_1;
create table if not exists koo.nil_temp0222_a2_1 as
select a.*
,case when b.nn is null then a.nn+3 else b.nn end as nn_end
from koo.nil_temp0222_a2 a 
left join koo.nil_temp0222_a2 b 
on a.userid = b.userid 
and a.nn1 = b.nn1 - 1;

select a.*,b.nn1 as session_id
from
(
  select userid,inserttime
    ,row_number() over(partition by userid order by inserttime asc) nn
  from koo.nil_temp0222 
  where userid = 1900000169
) a
left join koo.nil_temp0222_a2_1 b 
on a.userid = b.userid
and a.nn>=b.nn
and a.nn<b.nn_end
```

订单表，字段有订单编号和时间；取每月最后一天的最后三笔订单

```sql
select *
from 
(
  select *
  ,rank() over(partition by mm order by dd desc) as nn1
  ,row_number() over(partition by mm,dd order by inserttime desc) as nn2
  from
  (select cast(right(to_date(inserttime),2) as int) as dd,month(inserttime) as mm,userid,inserttime
  from koo.nil_temp0222) aa 
) bb 
where nn1 = 1 and nn2<=3;
```

数据库表Tourists，记录了某个景点7月份每天来访游客的数量，id字段刚好等于日期里面的几号。  现在请筛选出连续三天都有大于100天的日期。

```sql
select a.*,b.num as num2,c.num as num3
from table  a 
left join table b
on a.userid = b.userid
and a.dt = date_add(b.dt,-1)
left join table c
on a.userid = c.userid
and a.dt = date_add(c.dt,-2)
where b.num>100
and a.num>100
and c.num>100
```

现有A表，有21个列，第一列id，剩余列为特征字段，另外一个表B称为模式表，和A表结构一样，请找到A表中的特征符合B表中模式的数据，并记录下相对应的id

```sql
select aa.*
from 
(
  select *,concat(d1,d2,d3……d20) as mmd
  from table
) aa 
left join 
(
  select id,concat(d1,d2,d3……d20) as mmd
  from table
) bb 
on aa.id = bb.id
and aa.mmd = bb.mmd

--另一种方法
select a.*,sum(d1_jp,d2_jp……,d20_jp) as same_judge
from 
(
  select a.*
  ,case when a.d1 = b.d1 then 1 else 0 end as d1_jp
  ,case when a.d2 = b.d2 then 1 else 0 end as d2_jp
  ,case when a.d3 = b.d3 then 1 else 0 end as d3_jp
  ,case when a.d4 = b.d4 then 1 else 0 end as d4_jp
  ,case when a.d5 = b.d5 then 1 else 0 end as d5_jp
  ,case when a.d6 = b.d6 then 1 else 0 end as d6_jp
  ,case when a.d7 = b.d7 then 1 else 0 end as d7_jp
  ,case when a.d8 = b.d8 then 1 else 0 end as d8_jp
  ,case when a.d9 = b.d9 then 1 else 0 end as d9_jp
  ,case when a.d10 = b.d10 then 1 else 0 end as d10_jp
  ,case when a.d20 = b.d20 then 1 else 0 end as d20_jp
  ,case when a.d11 = b.d11 then 1 else 0 end as d11_jp
  ,case when a.d12 = b.d12 then 1 else 0 end as d12_jp
  ,case when a.d13 = b.d13 then 1 else 0 end as d13_jp
  ,case when a.d14 = b.d14 then 1 else 0 end as d14_jp
  ,case when a.d15 = b.d15 then 1 else 0 end as d15_jp
  ,case when a.d16 = b.d16 then 1 else 0 end as d16_jp
  ,case when a.d17 = b.d17 then 1 else 0 end as d17_jp
  ,case when a.d18 = b.d18 then 1 else 0 end as d18_jp
  ,case when a.d19 = b.d19 then 1 else 0 end as d19_jp
  from table a 
  left join table b 
  on a.id = b.id 
) aa
where sum(d1_jp,d2_jp……,d20_jp) = 19
```

我们把用户对商品的评分用稀疏向量表示，保存在数据库表t里面：

- t的字段有：uid，goods_id，star。uid是用户id  

- goodsid是商品id  

- star是用户对该商品的评分，值为1-5

计算向量两两之间的内积：对于两个不同的用户，如果他们都对同样的一批商品打了分，那么对于这里面的每个人的分数乘起来，并对这些乘积求和。

```sql
select aa.uid1,aa.uid2
,sum(star_multi) as result
from 
(
  select a.uid as uid1
  ,b.uid as uid2
  ,a.goods_id
  ,a.star * b.star as star_multi
  from t a 
  left join t b 
  on a.goods_id = b.goods_id
  and a.udi<>b.uid  
) aa 
group by 1,2   

--另一种方法
select uid1,uid2,sum(multiply) as result
from
(select t.uid as uid1, t.uid as uid2, goods_id,a.star*star as multiply
from a left join b 
on a.goods_id = goods_id
and a.uid<>uid) aa
group by goods
```

给出一堆数和频数的表格，统计这一堆数中位数

```sql
select a.*
,b.s_mid_n
,c.l_mid_n
,avg(b.s_mid_n,c.l_mid_n)
from 
(
  select 
  case when mod(count(*),2) = 0 then count(*)/2 else (count(*)+1)/2 end as s_mid
  ,case when mod(count(*),2) = 0 then count(*)/2+1 else (count(*)+1)/2 end as l_mid  
  from table 
) a 
left join 
(
  select id,num,row_number() over(partition by id order by num asc) nn
  from table
) b 
on a.s_mid = b.nn
left join 
(
  select id,num,row_number() over(partition by id order by num asc) nn
  from table
) c  
on a.l_mid = c.nn
```

表order有三个字段，店铺ID，订单时间，订单金额；查询一个月内每周都有销量的店铺

```sql
select distinct credit_level
from 
(
  select credit_level,count(distinct nn) as number
  from
  (
    select userid,credit_level,inserttime,month(inserttime) as mm
    ,weekofyear(inserttime) as week
    ,dense_rank() over(partition by credit_level,month(inserttime) order by weekofyear(inserttime) asc) as nn
    from koo.nil_temp0222 
    where substring(inserttime,1,7) = '2019-12'
    order by credit_level ,inserttime  
  ) aa 
  group by 1  
) bb
where number = (select count(distinct weekofyear(inserttime))
from koo.nil_temp0222 
where substring(inserttime,1,7) = '2019-12')
```

手写"连续活跃登陆"等类似场景的sql

```sql
# 该题目有不同的解法，可以使用row_number窗口函数或者使用lag函数
#  第一种解决方案，使用lag(向前)或者lead(向后)
 select
   *
 from 
 (
   select 
     user_id,
     date_id,
    lead(date_id) over(partition by user_id order by date_id) as last_date_id
  from 
  (
    select 
      user_id,
      date_id
    from wedw_dw.tmp_log
    where date_id>='2020-08-10'
      and user_id is not null  
      and length(user_id)>0
    group by user_id,date_id
    order by user_id,date_id
  )t
)t1
where datediff(last_date_id,date_id)=1
-- 第二种解决方案,使用row_number
select 
  user_id,
  min(date_id),
  max(date_id),
  count(1)
from 
(
  select
  t1.user_id
  ,t1.date_id
  ,date_sub(t1.date_id,rn) as dis
  from 
  (
    select 
      user_id,
      date_id,
      row_number() over(partition by user_id order by date_id asc) rn
    from 
    (
      select 
        user_id,
        date_id
      from wedw_dw.tmp_log
      where date_id>='2020-08-10'
        and user_id is not null  
        and length(user_id)>0
      group by user_id,date_id
      order by user_id,date_id
    )t
  )t1
)t2
group by user_id,dis 
having count(1)>2
```
