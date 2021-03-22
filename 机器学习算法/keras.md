#### 准备数据

#### 定义模型

#### 模型训练

##### 回调函数

`ReduceLROnPlateau`在损失指标停止改善或达到稳定时降低学习率。

```python
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
model.fit(X_train, Y_train, callbacks=[reduce_lr])
#受一个列表，因此可以安排多个回调。

from keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr): #定义回调schedule
   if lr < 0.001: return lr * 1.5 #如果lr太小,增加lr
   elif epoch < 5: return lr #前五个epoch，不改变lr
   elif epoch < 10: return lr * tf.math.exp(-0.1) #第五到第十个epoch，减少lr
   else: return lr * tf.math.exp(-0.05) #第10个epoch之后，减少量更少
  
callback = LearningRateScheduler(scheduler) #创建回调对象
model.fit(X_train, y_train, epochs=15, callbacks=[callback])
```

`EarlyStopping`可以非常有助于防止在训练模型时产生额外的冗余运行。`TerminateOnNaN`有助于防止在训练中产生梯度爆炸问题，因为输入`NaN`会导致网络的其他部分发生爆炸。

`ModelCheckpoint`可以以某种频率保存模型的权重

#### 评估模型

#### 模型预测

#### 模型保存

### $\text{Pytorch}$

张量是一种包含某种标量类型的 n 维数据结构。我们可以将张量看作是由一些数据构成的，还有一些元数据描述了张量的大小、所包含的元素的类型、张量所在的设备。但要在我们的计算机中表示它，我们必须为它们定义某种物理表示方法。最常用的表示方法是在内存中相邻地放置张量的每个元素，即将每一行写出到内存。为了记住张量的实际维度，我们必须将规模大小记为额外的元数据。

我该如何将这个逻辑位置转译为物理内存中的位置？步幅能让我们做到这一点：要找到一个张量中任意元素的位置，我将每个索引与该维度下各自的步幅相乘，然后将它们全部加到一起。

X[1, :] 就能得到这一行。重要的是：当我这样做时，不会创建一个新张量；而是会返回一个基于底层数据的不同域段（view）的张量。这意味着，如果我编辑该视角下的这些数据，它就会反映在原始的张量中。我们只需要记录一个说明该张量的数据位于顶部以下 2 个位置的偏移量（offset）。

##### 距离

###### 欧几里得距离

$$
E(p,q) = \sqrt{\sum_{i=1}^n(p_i-q_i)^2}
$$

```python
from math import *
def euclidean_distance(x, y):    
    return sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))
```

###### 标准化欧式距离

$$
D(p,q) = \sqrt{\sum_{i=1}^n(\frac{p_i-q_i}{s_i})^2}
$$

```python
def normalized_euclidean(a, b):
    sumnum = 0
    for i in range(len(a)):
        avg = (a[i] - b[i]) / 2
        si = ((a[i] - avg) ** 2 + (b[i] - avg) ** 2) ** 0.5
        sumnum += ((a[i] - b[i]) / si) ** 2
    return sumnum ** 0.5
```

###### 曼哈顿距离

$$
D(p,q) = \sum_{i=1}^n|p_i-q_i|
$$

```python
from math import *

def manhattan_distance(x,y):
    return sum(abs(a-b) for a,b in zip(x,y))
```

###### 汉明距离

```python
def hamming_distance(s1, s2):
    """Return the Hamming distance between equal-length sequences"""
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))
```

###### 赛切比雪夫距离

$$
D(p,q) = \max(|p_i-q_i|)
$$

```python
def chebyshev_distance(p, q):
    assert len(p) == len(q)
    return max([abs(x - y) for x, y in zip(p, q)])
def chebyshev_distance_procedural(p, q):
    assert len(p) == len(q)
    d = 0
    for x, y in zip(p, q):
        d = max(d, abs(x - y))
    return d
```

###### 马氏距离

$$
D(p,q) = \sqrt{(\vec{p}-\vec{q})^T\Sigma^{-1}(\vec{p}-\vec{q})}
$$

如果协方差矩阵为单位矩阵，马氏距离就简化为欧氏距离；如果协方差矩阵为对角阵，其也可称为正规化的欧氏距离。

```python
import pandas as pd
import scipy as sp
from scipy.spatial.distance import mahalanobis
datadict = {
'country': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Ecuador', 'Colombia', 'Paraguay', 'Peru', 'Venezuela'],
'd1': [0.34, -0.19, 0.37, 1.17, -0.31, -0.3, -0.48, -0.15, -0.61],
'd2': [-0.57, -0.69, -0.28, 0.68, -2.19, -0.83, -0.53, -1, -1.39],
'd3': [-0.02, -0.55, 0.07, 1.2, -0.14, -0.85, -0.9, -0.47, -1.02],
'd4': [-0.69, -0.18, 0.05, 1.43, -0.02, -0.7, -0.72, 0.23, -1.08],
'd5': [-0.83, -0.69, -0.39, 1.31, -0.7, -0.75, -1.04, -0.52, -1.22],
'd6': [-0.45, -0.77, 0.05, 1.37, -0.1, -0.67, -1.4, -0.35, -0.89]}
pairsdict = {
'country1': ['Argentina', 'Chile', 'Ecuador', 'Peru'],
'country2': ['Bolivia', 'Venezuela', 'Colombia', 'Peru']}
#DataFrame that contains the data for each country
df = pd.DataFrame(datadict)
#DataFrame that contains the pairs for which we calculate the Mahalanobis distance
pairs = pd.DataFrame(pairsdict)
#Add data to the country pairs
pairs = pairs.merge(df, how='left', left_on=['country1'], right_on=['country'])
pairs = pairs.merge(df, how='left', left_on=['country2'], right_on=['country'])
#Convert data columns to list in a single cell
pairs['vector1'] = pairs[['d1_x','d2_x','d3_x','d4_x','d5_x','d6_x']].values.tolist()
pairs['vector2'] = pairs[['d1_y','d2_y','d3_y','d4_y','d5_y','d6_y']].values.tolist()
mahala = pairs[['country1', 'country2', 'vector1', 'vector2']]
#Calculate covariance matrix
covmx = df.cov()
invcovmx = sp.linalg.inv(covmx)
#Calculate Mahalanobis distance
mahala['mahala_dist'] = mahala.apply(lambda x: (mahalanobis(x['vector1'], x['vector2'], invcovmx)), axis=1)
mahala = mahala[['country1', 'country2', 'mahala_dist']]
```

###### 兰氏距离

$$
D(\vec{p},\vec{q}) = \sum_{i=1}^n\frac{|p_i-q_i|}{|p_i|+|q_i|}
$$

```python
def canberra_distance(p, q):
    n = len(p)
    distance = 0
    for i in n:
        if p[i] == 0 and q[i] == 0:
            distance += 0
        else:
            distance += abs(p[i] - q[i]) / (abs(p[i]) + abs(q[i]))
    return distance
```

###### 闵科夫斯基距离

$$
D(\vec{p},\vec{q})=(\sum_{i=1}^n|p_i-q_i|^p)^{\frac{1}{p}}
$$

```
def minkowski_distance(p, q, n):
    assert len(p) == len(q)
    return sum([abs(x - y) ^ n for x, y in zip(p, q)]) ^ 1 / n
def minkowski_distance_procedural(p, q, n):
    assert len(p) == len(q)
    s = 0
    for x, y in zip(p, q):
        s += abs(x - y) ^ n
    return s ^ (1 / n)
```

闵氏距离的缺点主要有两个：将各个分量的量纲，也就是“单位”当作相同看待了；没有考虑各个分量的分布（期望，方差等)可能是不同的

###### 编辑距离

编辑距离是指两个字串之间，由一个转成另一个所需的最少编辑操作次数。编辑操作包括将一个字符替换成另一个字符，插入一个字符，删除一个字符。一般来说，编辑距离越小，两个串的相似度越大。

莱文斯坦比：$r=\frac{S-l}{S}$，其中$S$是指$s_1$和$s_2$字串的长度总和，$l$是类编辑距离。类编辑距离，在类编辑距离中删除、插入依然+1，但是替换+2。

###### 余弦相似度

$$
\cos(\vec{p},\vec{q}) = \frac{\sum_{i=1}^n p_i\times q_i}{\sqrt{\sum p_i^2}\times\sqrt{\sum q_i^2}}
$$

```python
from math import *

def square_rooted(x):
    return round(sqrt(sum([a*a for a in x])),3)

def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
   denominator = square_rooted(x)*square_rooted(y)
   return round(numerator/float(denominator),3)
```

###### 杰卡德相似度

两个集合A和B交集元素的个数在A、B并集中所占的比例，称为这两个集合的杰卡德系数
$$
J(A,B) = \frac{A\cap B}{A\cup B}
$$

```python
def jaccard_sim(a, b):
    unions = len(set(a).union(set(b)))
    intersections = len(set(a).intersection(set(b)))
    return intersections / unions
```

###### 杰卡德距离

$$
J_{\delta}=1-J(A,B)
$$

```python
def jaccard_similarity(x,y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)
```

###### Dice系数

$$
S(A,B) = \frac{2|A\cap B}{|A|+|B|}
$$

```python
def dice_coefficient(a, b):
    """dice coefficient 2nt/na + nb."""
    a_bigrams = set(a)
    b_bigrams = set(b)
    overlap = len(a_bigrams & b_bigrams)
    return overlap * 2.0/(len(a_bigrams) + len(b_bigrams))
```

特征间的单位（尺度）可能不同，为了消除特征间单位和尺度差异的影响，以对每维特征同等看待，需要对特征进行归一化。

原始特征下，因尺度差异，其损失函数的等高线图可能是椭圆形，梯度方向垂直于等高线，下降会走zigzag路线，通过对特征进行zero-mean and unit-variance变换后，其损失函数的等高线图更接近圆形，梯度下降的方向震荡更小，收敛更快，

- **减一个统计量**可以看成**选哪个值作为原点，是最小值还是均值，并将整个数据集平移到这个新的原点位置**。如果特征间偏置不同对后续过程有负面影响，则该操作是有益的，可以看成是某种**偏置无关操作**；如果原始特征值有特殊意义，比如稀疏性，该操作可能会破坏其稀疏性。
- **除以一个统计量**可以看成在**坐标轴方向上对特征进行缩放**，用于**降低特征尺度的影响，可以看成是某种尺度无关操作**。缩放可以使用最大值最小值间的跨度，也可以使用标准差（到中心点的平均距离），前者对outliers敏感，outliers对后者影响与outliers数量和数据集大小有关，outliers越少数据集越大影响越小。
- **除以长度**相当于把长度归一化，**把所有样本映射到单位球上**，可以看成是某种**长度无关操作**，比如，词频特征要移除文章长度的影响，图像处理中某些特征要移除光照强度的影响，以及方便计算余弦距离或内积相似度等。

与距离计算无关的概率模型，不需要feature scaling，比如Naive Bayes；与距离计算无关的基于树的模型，不需要feature scaling

###### 损失函数

reduction-三个值，none: 不使用约简；mean:返回loss和的平均值；sum:返回loss的和。默认：mean。

| 名称                | 定义                                   | 函数                                                         |
| ------------------- | -------------------------------------- | ------------------------------------------------------------ |
| `L1`损失            |                                        | `torch.nn.L1Loss(reduction='mean')`                          |
|                     |                                        | `torch.nn.MSELoss(reduction='mean')`                         |
| 交叉熵损失          |                                        | `torch.nn.CrossEntropyLoss(weight=None, reduction='mean')`   |
| `KL`散度损失        |                                        | `torch.nn.KLDivLoss(reduction='mean')`                       |
| 二进制交叉损失      |                                        | `torch.nn.BCELoss(weight=None, reduction='mean')`            |
| `BCEWithLogitsLoss` |                                        | `torch.nn.BCEWithLogitsLoss(weight=None, reduction='mean', pos_weight=None)` |
| `MarginRankingLoss` | $loss(x,y)=max(0,-y*(x_1-x_2))+margin$ | `torch.nn.MarginRankingLoss(margin=0.0, reduction='mean')`   |
|                     |                                        |                                                              |
|                     |                                        |                                                              |
|                     |                                        |                                                              |
|                     |                                        |                                                              |
|                     |                                        |                                                              |
|                     |                                        |                                                              |
|                     |                                        |                                                              |
|                     |                                        |                                                              |
|                     |                                        |                                                              |
|                     |                                        |                                                              |
|                     |                                        |                                                              |
|                     |                                        |                                                              |

