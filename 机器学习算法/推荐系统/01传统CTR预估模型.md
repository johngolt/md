记“某时某刻某地用户看到某个广告”这个事件为E。事件E发生后的结果只能是点击或不点击两种情况，我们用`Bernoulli`分布来刻画，对应参数为$\theta_E(x)$，其中$x$表示事件$E$的所有背景知识，包括用户信息、广告信息、场景及上下文信息。点击的概率$P_E(y=1|x)=\theta_E(x)$。微观视角上看来每个事件E是无关联的，伯努利分布的参数$\theta_E(x)$仅靠对事件E的一次抽样无法准确学习，即：微观层面的CTR不可被准确预测。

目前做法是假设$\theta_E(x)$跟单个事件E本身无关，仅跟事件的特征有关，即：$\theta_E(x)=\theta(x)$。对于Bernoulli分布而言，其期望值即为参数$\theta$。因此在上述假设下我们可以进一步得到：$p(y=1|x)=\theta(x)$。$\theta$是一个跟$x$有关的常量。

`CTR`和推荐算法的本质区别是：CTR最终预测的是那个有物理意义的数值`CTR`；推荐算法最终产生的是一个推荐列表，是一个item的相对位置。

CTR即点击通过率，指网络广告的点击到达率，即该广告的实际点击次数除以广告的展现量。

在 `cost-per-click:CPC` 广告中广告主按点击付费。为了最大化平台收入和用户体验，广告平台必须预测广告的 `CTR` ，称作 `predict CTR: pCTR` 。对每个用户的每次搜索`query`，有多个满足条件的广告同时参与竞争。只有 `pCTR x bid price` 最大的广告才能竞争获胜，从而最大化 `eCPM` ：$\text{eCPM}=\text{pCTR}\times \text{bid price}$。

基于最大似然准则可以通过广告的历史表现得统计来计算 `pCTR` 。假设广告曝光了 100次，其中发生点击 5 次，则 `pCTR = 5%`。其背后的假设是：忽略表现出周期性行为或者不一致行为的广告，随着广告的不断曝光每个广告都会收敛到一个潜在的真实点击率

广告被点击的概率取决于两个因素：广告被浏览的概率、广告浏览后被点击的概率。因此有：
$$
P(\text{click}|ad, pos) = p(\text{click}|ad, pos, seen)\times p(seen|ad, pos)
$$
假设：在广告被浏览到的情况下，广告被点击的概率与其位置无关，仅与广告内容有关。广告被浏览的概率与广告内容无关，仅与广告位置有关。则有：
$$
P(\text{click}|ad, pos) = p(\text{click}|ad, seen)\times p(seen|pos)
$$
第一项$p(\text{click}|ad, seen)$就是我们关注和预测的 `CTR` 。第二项与广告无关，是广告位置的固有属性。可以通过经验来估计这一项：统计该广告位的总拉取次数impress，以及总曝光次数seen，则：$p(seen|pos)=\frac{seen}{impress}$，这也称作广告位的曝光拉取比。广告可能被拉取（推送到用户的页面），但是可能未被曝光（未被用户浏览到）。

数据集构建：样本的特征从广告基本属性中抽取，将每个广告的真实点击率 `CTR` 来作为 `label` 。考虑到真实点击率 `CTR` 无法计算，因此根据每个广告的累计曝光次数、累计点击次数从而得到其经验点击率$\overline{\text{CTR}}$来作为 `CTR` 

##### `LR`模型

提出利用 `LR` 模型来预测新广告的`CTR`。将 `CTR` 预估问题视作一个回归问题，采用逻辑回归 `LR` 模型来建模，因为 `LR` 模型的输出是在 `0` 到 `1` 之间。
$$
\text{pCTR} = \frac{1}{1+\exp(-\sum_{i}\omega_i\times f_i)}
$$
其中$f_i$表示从广告中抽取的第$i$个特征，$\omega_i$为该特征对应的权重。

模型的损失函数为交叉熵：$\mathcal{L}= -[\text{pCTR}\times \log(\overline{\text{CTR}})]+(1-\text{pCTR})\times \log(1-\overline{\text{CTR}})$

采取了一些通用的特征预处理方法：模型添加了一个`bias feature`，该特征的取值恒定为 1。对于每个特征$f_i$，人工构造额外的两个非线性特征：$\log(f_i+1), f_i^2$。对所有特征执行标准化，标准化为均值为0、方差为1 。对所有特征执行异常值截断：对于每个特征，任何超过均值 5 个标准差的量都被截断。

###### 评价指标

评估指标测试集上每个广告的 `pCTR` 和真实点击率的平均 `KL` 散度。
$$
\overline{\mathbb{D}_{KL}}=\frac{1}{T}\sum_{i=1}^{T}(\text{pCTR}(ad_i)\times \log\frac{\text{pCTR}(ad_i)}{\overline{\text{CTR}}(ad_i)}+(1-\text{pCTR}(ad_i))\times \log\frac{1-\text{pCTR}(ad_i)}{1-\overline{\text{CTR}}(ad_i)})
$$
`KL` 散度衡量了$\text{pCTR}$和真实点击率之间的偏离程度。一个理想的模型，其 `KL` 散度为 0 ，表示预估点击率和真实点击率完全匹配。

##### `POLY2`模型

`LR` 模型只考虑特征之间的线性关系，而`POLY2` 模型考虑了特征之间的非线性关系。捕获非线性特征的一个常用方法是采用核技巧，如高斯核`RBF`，将原始特征映射到一个更高维空间。在这个高维空间模型是线性可分的，即：只需要考虑新特征之间的线性关系。但是核技巧存在计算量大，内存需求大的问题。

`POLY2`通过多项式映射 `polynomially mapping` 数据的方式来提供非线性特征，在达到接近核技巧效果的情况下大幅度降低内存和计算量。设低维样本空间为$n$维度，低维样本$\vec{\mathbf{x}}=(x_1,\cdots,x_n)^T$。如果不用核技巧，仅考虑使用一个多项式映射，则我们得到
$$
\phi(\vec{\mathbf{x}}) = [1,x_1,\cdots,x_n,x_1^2,\cdots,x_n^2,x_1x_2,\cdots,x_{n-1}x_n]^T
$$
结合`LR` 模型，则得到 `POLY2` 模型：
$$
z(\vec{\mathbf{x}})=\omega_0+\sum_{i=1}^p\omega_ix_i+\sum_{i=1}^{K}\sum_{j=i+1}^{K}\omega_{ij}\times x_i\times x_j\\
y(\vec{\mathbf{x}})=\frac{1}{1+\exp(-z(\vec{\mathbf{x}}))}
$$

`POLY2` 模型的优点：除了线性特征之外，还能够通过特征组合自动捕获二阶特征交叉产生的非线性特征。

缺点：参数太多导致计算量和内存需求发生爆炸性增长；数据稀疏导致二次项参数训练困难，非常容易过拟合。参数$\omega_{i,j}$的训练需要大量的$x_i,x_j$都非零的样本。而大多数应用场景下，原始特征本来就稀疏，特征交叉之后更为稀疏。这使得训练$\omega_{i,j}$的样本明显不足，很容易发生过拟合。

##### `FM`模型

推荐系统面临的问题是评分预测问题。给定用户集合$\mathbb{U}=\{u_1,\cdots,u_M\}$、物品集合$\mathbb{I}=\{i1,\cdots,i_N\}$，模型是一个评分函数：$f:\mathbb{U}\times\mathbb{I}\to\mathbb{R}$。事实上除了已知部分用户在部分物品上的评分之外，通常还能够知道一些有助于影响评分的额外信息。对每一种上下文，我们用变量$c\in\mathbb{C}$来表示，$\mathbb{C}$为该上下文的取值集合。假设所有的上下文为$\mathbb{C}_3,\cdots,\mathbb{C}_K$，则模型为：
$$
f:\mathbb{U}\times\mathbb{I}\times\mathbb{C}_3\times\cdots\times \mathbb{C}_K\to\mathbb{R}
$$
所有离散特征都经过特征转换。

![](../../picture/1/410.png)

上下文特征 `context` 类似属性 `property` 特征，它和属性特征的区别在于：属性特征是作用到整个用户（用户属性）或者整个物品（物品属性），其粒度是用户级别或者物品级别。上下文特征是作用到用户的单个评分事件上，粒度是事件级别，包含的评分信息更充足。事实上属性特征也称作静态画像，上下文特征也称作动态画像。业界主流的做法是：融合静态画像和动态画像。另外，业界的经验表明：动态画像对于效果的提升远远超出静态画像。

和 `POLY2` 相同`FM` 也是对二路特征交叉进行建模，但是`FM` 的参数要比 `POLY2` 少得多。将样本为：$\vec{\mathbf{x}}=(x_1,x_2,\cdots,x_K)^T$。则 `FM` 模型为：
$$
\hat{y}(\mathbf{x})=\omega_0+\sum_{i=1}^K\omega_ix_i+\sum_{i=1}^{K}\sum_{j=i+1}^{K}\hat{w}_{i,j}x_ix_j
$$
其中$\hat{w}_{i,j}$是交叉特征的参数，它由一组参数定义：$\hat{w}_{i,j}=<\vec{\mathbf{v}}_i,\vec{\mathbf{v}}_j>=\sum_{l=1}^dv_{i,l}\times v_{j,l}$
$$
\hat{\mathbf{W}} = \left[\begin{array}{cccc}\hat{w}_{1,1}&\hat{w}_{1,2}&\cdots&\hat{w}_{1,K}\\
\hat{w}_{2,1}&\hat{w}_{2,2}&\cdots&\hat{w}_{2,K}\\
\cdot&\cdot&\cdots&\cdot\\
\hat{w}_{K,1}&\hat{w}_{K,2}&\cdots&\hat{w}_{K,K}\end{array}\right]=\mathbf{V}^T\mathbf{V}=\left[\begin{array}{cccc}\vec{\mathbf{v}}_1^T\\
\vec{\mathbf{v}}_2^T\\
\cdot\\ \cdot \\ \cdot\\\vec{\mathbf{v}}_K^T
\end{array}\right]\left[\begin{array}{cccc}\vec{\mathbf{v}}_1^T  &  \vec{\mathbf{v}}_2^T&\cdots&\vec{\mathbf{v}}_K^T
\end{array}\right]
$$
模型待求解的参数为：$\omega_0\in \mathbb{R},\vec{\mathbf{w}}\in \mathbb{R}^n, \mathbf{V}=(\vec{\mathbf{v}}_i,\cdots,\vec{\mathbf{x}}_K)\in \mathbb{R}^{d\times K}$。

其中：$\omega_0$表示全局偏差，$\omega_i$用于捕捉第$i$个特征和目标之间的关系；$\hat{\omega}_{i,j}$用于捕捉$(i,j)$二路交叉特征和目标之间的关系。$\vec{\mathbf{v}}_i$代表特征$i$的`representation vector`

对于每个输入特征$x_i$，模型都需要学习一个低维的隐向量表达$\mathbf{v}_i$。
$$
\\
\begin{array}{l}\sum_{i=1}^{K}\sum_{j=i+1}^{K}\hat{w}_{i,j}x_ix_j &=\sum_{i=1}^{K}\sum_{j=i+1}^{K}(\sum_{l=1}^dv_{il}v_{j_l})x_ix_j\\
&=\sum_{l=1}^d[\sum_{i=1}^{K}\sum_{j=i+1}^{K}(v_{il}x_i)(v_{jl}x_j)]\\
&=\sum_{l=1}^d\frac{1}{2}[(\sum_{i=1}^{K}v_{il}x_i)^2-\sum_{i=1}^{K}(v_{il}x_i)^2]
\end{array}
$$
因此有：$\hat{y}(\vec{\mathbf{x}})=\omega_0+\sum_{i=1}^K\omega_ix_i+\sum_{l=1}^d\frac{1}{2}[(\sum_{i=1}^{K}v_{il}x_i)^2-\sum_{i=1}^{K}(v_{il}x_i)^2]$

`FM` 模型可以用于求解分类问题 ，也可以用于求解回归问题。

- 对于回归问题，其损失函数为`MSE` 均方误差

$$
\mathcal{L} = \sum_{(\vec{\mathbf{x}},y)\in\mathbb{S}}(\hat{y}(\vec{\mathbf{x}})-y)^2+\sum_{\theta\in\Theta}\lambda_\theta\times\theta^2
$$

- 对于二分类问题，其损失函数为交叉熵：

$$
\begin{array}{c}
\phi(\vec{\mathbf{x}})=\omega_0+\sum_{i=1}^K\omega_ix_i+\sum_{l=1}^d\frac{1}{2}[(\sum_{i=1}^{K}v_{il}x_i)^2-\sum_{i=1}^{K}(v_{il}x_i)^2]\
p(\hat{y}=y|\vec{\mathbf{x}})=\frac{1}{1+\exp(-y\phi(\vec{\mathbf{x}}))}\\
\mathcal{L} = -\sum_{(\vec{\mathbf{x}},y)\in\mathbb{S}}\log p(\hat{y}=y|\vec{\mathbf{x}}) + \sum_{\theta\in\Theta}\lambda_\theta\times\theta^2\end{array}
$$

其中$\mathbf{\Theta}=\{w_0,\vec{\mathbf{w}},\mathbf{V}\}$，其中$\lambda_\theta$为参数$\theta$的正则化系数

`FM` 模型可以处理不同类型的特征：

- 离散型特征 `categorical`：`FM` 对离散型特征执行 `one-hot` 编码。
- 离散集合特征 `categorical set`：`FM` 对离散集合特征执行类似 `one-hot` 的形式，但是执行样本级别的归一化。
- 数值型特征 `real valued`：`FM`直接使用数值型特征，不做任何编码转换。

在交叉特征高度稀疏的情况下，参数仍然能够估计。因为交叉特征的参数不仅仅依赖于这个交叉特征，还依赖于所有相关的交叉特征。这相当于增强了有效的学习数据；能够泛化到未被观察到的交叉特征。

###### `ALS`优化算法

$$
\hat{y}(\vec{\mathbf{x}};\mathbf{\Theta})=\omega_0+\sum_{i=1}^K\omega_ix_i+\sum_{l=1}^d\frac{1}{2}[(\sum_{i=1}^{K}v_{il}x_i)^2-\sum_{i=1}^{K}(v_{il}x_i)^2]
$$

对每个$\theta\in \mathbf{\Theta}=\{w_0,\vec{\mathbf{w}},\mathbf{V}\}$，可以将$\hat{y}$分解为$\theta$的线性部分和偏置部分：$\hat{y}(\vec{\mathbf{x}};\mathbf{\Theta})=h_\theta(\vec{\mathbf{x}})\times \theta +g_\theta(\vec{\mathbf{x}})$。其中$h_\theta(\vec{\mathbf{x}}),g_\theta(\vec{\mathbf{x}})$与$\theta$无关。

对于$w_0$有：
$$
h_{w_{0}}=1\\
g_{w_0} = \sum_{i=1}^K\omega_ix_i+\sum_{l=1}^d\frac{1}{2}[(\sum_{i=1}^{K}v_{il}x_i)^2-\sum_{i=1}^{K}(v_{il}x_i)^2]
$$
对于$w_i,i=1,2,\cdots,K$有：
$$
h_{w_i} = x_i\\
g_{w_i}=\omega_0+\sum_{j=1,j\ne i}^K\omega_jx_j+\sum_{l=1}^d\frac{1}{2}[(\sum_{j=1}^{K}v_{jl}x_j)^2-\sum_{j=1}^{K}(v_{jl}x_j)^2]
$$
对于$v_{i,l},i=1,2,\cdots,K;l=1,2.\cdots,d$有：
$$
h_{v_{i,l}} = \sum_{j=1,j\ne i}^Kv_{j,l}\times x_j\times x_i\\
g_{v_{i,l}} = \omega_0+\sum_{i=1}^K\omega_ix_i+\sum_{i^{\prime}=1}^K\sum_{j^{\prime}=i^{\prime}+1}^K\sum_{l^{\prime}=1,(l^{\prime},i^{\prime},j^{\prime})\ne (l,i,i)}v_{i^{\prime},l^{\prime}}\times v_{j^{\prime},l^{\prime}}\times x_{i^{\prime}}\times x_{j^{\prime}}
$$
考虑均方误差损失函数，最小值点的偏导数为 0 ，则有：
$$
\theta = -\frac{\sum_{(\vec{\mathbf{x}},y)\in\mathbb{S}}(g_\theta(\vec{\mathbf{x}})-y)\times h_\theta(\vec{\mathbf{x}})}{\sum_{(\vec{\mathbf{x}},y)\in\mathbb{S}}h^2_\theta(\vec{\mathbf{x}})+\lambda_\theta}
$$
`ALS` 通过多轮次、轮流迭代求解$\theta\in\mathbf{\Theta}$即可得到模型的最优解。在迭代之前初始化参数，其中：$w_0,\vec{\mathbf{w}}$通过零初始化，$\mathbf{V}$的每个元素通过均值为0、方差为$\sigma$的正太分布随机初始化。每一轮迭代时：首先求解$w_0,\vec{\mathbf{w}}$，因为相对于二阶交叉的高阶特征，低阶特征有更多的数据来估计其参数，因此参数估计更可靠。然后求解$\mathbf{V}$。这里按照维度优先的准确来估计：先估计所有`representation` 向量的第 `1` 维度，然后是第 `2` 维，... 最后是第 `d` 维。

`FM` 要优于 `POLY2` ，原因是：交叉特征非零的样本过于稀疏使得无法很好的估计 ；但是在 `FM` 中，交叉特征的参数可以从很多其它交叉特征中学习，使得参数估计更准确。

##### `FFM`模型

考虑一组特征：“性别、年龄、城市”。假设：“年龄”取值集合为 `[18,19,20]`， “城市” 取值集合为 `[北京,上海,广州,深圳]` 。把离散特征 `one-hot` 编码，设各 `binary` 特征分别记作：`male,female,age18,age19,age20,bj,sh,gz,sz`， `y` 表示样本标签

| 域   | 性别 |      | 年龄 |      |      |      | 价格 |      |       |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- |
| 特征 | 男   | 女   | 15   | 25   | 50   | >50  | 500  | 1000 | >1000 |
| 样本 | 1    | 0    | 0    | 0    | 1    | 0    | 0    | 1    | 0     |

在 `FM` 模型中，每个特征的表示向量只有一个。`FFM` 算法认为：`age=18` 和 `sh=1` 之间的区别，远远大于 `age=18` 和 `age=20` 之间的区别。因此，`FFM` 算法将特征划分为不同的域`field`。其中：特征`male=1, female=1`属于性别域 `gender field` 。特征`age18=1,age19=1,age20=1`属于年龄域 `age field` 。特征`bg=1,shz=1,gz=1,sz=1`属于城市域 `city field` 。`FFM` 中每个特征的表示向量有多个，用于捕捉该特征在不同`field` 中的含义。假设样本共有$F$个`field`，特征$x_i,x_j$分别属于域$f_i,f_j$，$x_i$有$F$个隐向量$(\mathbf{v}_{i,1},\cdots,\mathbf{v}_{i,F})$，$\mathbf{v}_i$是一个$d$维向量。
$$
\hat{y}(\vec{\mathbf{x}})=\omega_0+\sum_{i=1}^K\omega_ix_i+\sum_{i=1}^K\sum_{j=i+1}^K\hat{\omega}_{ij}\times x_i\times x_j\\
\hat{\omega}_{ij}=<\vec{\mathbf{v}}_{i,f_j},\vec{\mathbf{v}}_{j,f_i}>
$$

和 `FM` 相比，通常 `FFM` 中 `representation` 向量的维度要低的多。和 `FM` 相同，`FFM` 模型也可以用于求解分类问题，也可以用于求解回归问题。 

`FFM` 模型需要为每个特征分配一个 `field` ，离散型特征：通常对离散型特征进行 `one-hot` 编码，编码后的所有二元特征都属于同一个 `field` 。数值型特征：数值型特征有两种处理方式：不做任何处理，简单的每个特征分配一个`field` 。数值特征离散化之后，按照离散型特征分配 `field` 。离散集合特征`categorical set`：所有特征都属于同一个`field`，此时 ， `FFM` 退化为 `FM` 模型。

##### `GBDT+LR`

`GBDT-LR` 模型利用 `GBDT` 作为特征抽取器来抽取特征、利用 `LR` 作为分类器来执行分类预测。

###### 归一化熵

假设样本集合有$N$个样本，样本集合的经验`CTR` 为$\overline{p}$（它等于所有正类样本数量除以总样本数量）。

假设第$p$个样本预测为正类的概率为$p_i$，其真实标签为$y_i\in\{-1,+1\}$。定义背景点击率 `background CTR` 为样本集合经验 `CTR` ，它的熵定义为背景熵：
$$
H_{bg} = -[\overline{p}\log \overline{p}+(1-\overline{p})\log(1-\overline{p})]
$$
背景熵衡量了样本集合的类别不平衡程度，也间接的衡量了样本集合的预测难度。类别越不均衡预测难度越简单，因为只需要将所有样本预测为最大的类别即可取得非常高的准确率。定义模型在样本集合熵的损失函数为：
$$
\mathcal{L} = -\sum_{i=1}^N\left[\frac{1+y_i}{2}\log p_i+\frac{1-y_i}{2}\log(1-p_i)\right]
$$
每个样本的损失为交叉熵。定义归一化熵 `NE` 为：模型在所有样本的平均损失函数除以背景熵。
$$
\text{NE} = \frac{\mathcal{L}}{N}/H_{bg}
$$
在平均损失相同的情况下，样本集越不均衡则越容易预测，此时 `NE` 越低。

###### `GBDT`特征抽取

有两种最简单的特征转换方式：连续特征离散化：将连续特征的取值映射到一个个分散的分桶里，从而离散化；离散特征交叉：类似 `FM` 模型采用二路特征交叉（或者更高阶）来学习高阶非线性特征。

`GDT`将每棵子树视为一个离散特征，其叶结点的编号为特征的取值并执行 `one-hot` 编码。假设 `BDT` 有两棵子树，第一棵有 `3` 个叶结点，第二棵有`2` 个叶结点。则样本提取后有两个特征：第一个特征取值为 `{1,2,3}`，第二个特征取值为 `{1,2}` 。假设某个样本被划分到第一棵子树的叶结点 `2`，被划分到第二棵子树的叶结点 `1`，则它被转换后的特征为：`[0,1,0,1,0]`。其中：前三项对应于第一个离散特征的 `one-hot`，后两项对应于第二个离散特征的 `one-hot` 。



![](../../picture/1/303.png)

`GBDT-LR`采用梯度提升树来训练每棵子树，因此这种特征提取方式可以视为基于决策树的有监督特征编码：

- 它将一组实值向量 `real-valued vector` 转换为一组二元向量 `binary-valued vector` 。
- 每棵子树从根节点到叶节点的遍历表示某些特征转换规则。
- 在转换后的二元向量上拟合线性分类器本质上是学习每个规则的权重。

考虑数据新鲜度，我们需要用最新的样本更新模型。有两种更新策略：每天用最新的数据重新训练模型。每天或者每隔几天来训练特征提取器 `BDT` ，但是用最新的数据在线训练 `LR` 线性分类器。

###### 优化技巧

在 `GBDT-LR` 模型中，子树的数量越大模型表现越好，但是计算代价、内存代价越高。但是随着子树的增多，每增加一棵子树获得的效益是递减的。这就存在平衡：新增子树的代价和效益的平衡。

在 `GBDT-LR` 模型中，样本特征越大模型表现越好，但是计算代价、内存代价越高。但是随着特征的增多，尤其是无效特征的增多，每增加一个特征获得的效益是递减的。这就存在平衡：新增特征的代价和效益的平衡。

为衡量特征数量的影响，我们首先对特征重要性进行排序，然后考察 `topK` 重要性特征的效果。可以通过 `Boosting Feature Importance` 来衡量特征重要性。有三种度量方法（如 `XGBoolst/LightGBM` ）：

- `weight`：特征在所有子树中作为分裂点的总次数
- `gain`：特征在所有子树中作为分裂点带来的损失函数降低总数
- `cover`：特征在所有子树中作为分裂点包含的总样本数

###### 模型校准

模型校准分为两类：模型预测能力不足导致的校准；训练数据分布和线上数据分布不一致导致的校准

给定样本集$\mathbb{D}=\{(\vec{\mathbf{x}}_1,y_1)\cdots,(\vec{\mathbf{x}}_N,y_N)\}$，假设模型预估的 `pCTR` 分别为：$(\hat{y}_1,\cdots,\hat{y}_N)$。则样本集的经验 `CTR` 为：
$$
\overline{\text{CTR}} = \frac{\sum_{i=1}^N\mathbb{I}(y_i=1)}{N}
$$
样本集的预估平均 `CTR` 为：
$$
\overline{\text{CTR}}_{\text{pred}} =\frac{\sum_{i=1}^N\hat{y}_i}{N}
$$
定义校准系数为：预估平均 `CTR` 和经验 `CTR` 之比：
$$
\text{ratio} =\frac{\overline{\text{CTR}}_{\text{pred}}}{\overline{\text{CTR}}}
$$
它衡量了模型预期点击次数和实际观察到的点击次数之比，它的值与 1 的差异越小，则模型的表现越好。假设模型预估的结果为$\hat{y}$，则校准后的预估结果为：
$$
\hat{y}_{\text{new}} = \frac{\hat{y}}{\text{ratio}}
$$

##### `LS-PLM`模型

`LS-PLM` 模型基于分而治之的策略：首先将特征空间划分为几个局部区域；然后在每个区域中建立一个广义线性模型；最后将每个广义线性模型加权作为最终输出

`LS-PLM` 具有非线性、可扩展性、稀疏性的优点。

- 非线性：如果特征空间划分区域足够多，则 `LS-PLM` 模型将拟合任何复杂的非线性函数。
- 可扩展性：`LS-PLM` 可以扩展到超大规模和超高维特征。
- 稀疏性：带有$\text{L}_1$和$\text{L}_{2,1}$正则化的 `LS-PLM` 模型具有很好的稀疏性。

给定数据集$\mathbb{D}=\{(\vec{\mathbf{x}}_1,y_1)\cdots,(\vec{\mathbf{x}}_N,y_N)\}$，`LS-PLM` 算法基于分而治之的策略，将整个特征空间划分为一些局部区域，对每个区域采用广义线性分类模型：
$$
p(y=1|\vec{\mathbf{x}})=g(\sum_{j=1}^m\sigma(\vec{\mathbf{u}}_j\cdot\vec{\mathbf{x}})\times\eta(\vec{\mathbf{w}}_j\cdot\vec{\mathbf{x}}))
$$
其中：$\Theta=\{\vec{\mathbf{u}}_1,\cdots,\vec{\mathbf{u}}_m,\vec{\mathbf{w}}_1,\cdots,\vec{\mathbf{w}}_m\},\vec{\mathbf{u}}_j\in\mathbb{R}^d,\vec{\mathbf{w}}_j\in\mathbb{R}^d$为模型参数。

因此$\Theta$有两种表示方法：列向量：$\Theta=(\vec{\theta}_1,\cdots,\vec{\theta}_{2m})$。其中：
$$
\vec{\theta}_j=\left\{\begin{array}{ll}{\vec{\mathbf{u}}_j} & {1\le j\le m} \\ {\vec{\mathbf{w}}_{j-m}} & {m+1\le j\le 2m}\end{array}\right.
$$
行向量：$\Theta=(\vec{\Theta}^1,\cdots,\vec{\Theta}^{d})^T$。其中：$\vec{\Theta}^i=(u_{1,i},\cdots,u_{m,i},w_{1,i},\cdots,w_{m,i})^T$

$\sigma(\vec{\mathbf{u}}_j\cdot\vec{\mathbf{x}})$将样本划分到$m$个区域，$\eta(\vec{\mathbf{w}}_j\cdot\vec{\mathbf{x}}))$对每个空间进行预测，$g(\cdot)$用于对结果进行归一化。一种简单的情形是：
$$
\sigma(\vec{\mathbf{u}}_j\cdot\vec{\mathbf{x}})=\frac{\exp(\vec{\mathbf{u}}_j\cdot\vec{\mathbf{x}})}{\sum_{i=1}^m\exp(\vec{\mathbf{u}}_j\cdot\vec{\mathbf{x}})},\quad\eta(\vec{\mathbf{w}}_j\cdot\vec{\mathbf{x}}))=\frac{1}{1+\exp(-\vec{\mathbf{w}}_j\cdot\vec{\mathbf{x}})},\quad g(z)=z
$$
此时有：
$$
p(y=1|\vec{\mathbf{x}})=\frac{\exp(\vec{\mathbf{u}}_j\cdot\vec{\mathbf{x}})}{\sum_{i=1}^m\exp(\vec{\mathbf{u}}_j\cdot\vec{\mathbf{x}})}\times \frac{1}{1+\exp(-\vec{\mathbf{w}}_j\cdot\vec{\mathbf{x}})}
$$
这可以被认为是一种混合模型：
$$
p(y=1|\vec{\mathbf{x}})=\sum_{j=1}^mp(z=j|\vec{\mathbf{x}})\times p(y=1|z=j,\vec{\mathbf{x}})
$$
其中：$p(z=j|\vec{\mathbf{x}})$表示样本划分到区域$j$的概率。$p(y=1|z=j,\vec{\mathbf{x}})$表示在区域$j$中，样本$\vec{\mathbf{x}}$属于正类的概率。`LS-PLM` 模型的目标函数为：
$$
\mathcal{J}=\text{loss}(\Theta)+\lambda||\Theta||_{2,1}+\beta||\Theta||_1
$$
其中：`loss` 是负的对数似然损失函数。$||\Theta||_{2,1}$和$||\Theta||_{1}$是$\text{L}_{2,1}$和$\text{L}_{1}$正则化项。该正则化先计算每个维度在各区域的正则化，然后将所有维度的正则化直接累加。$||\Theta||_{2,1}$正则化主要用于特征选择；$||\Theta||_{1}$主要用于模型稀疏性，但是它也有助于特征选择。

由于$\text{L}_{2,1}$和$\text{L}_{1}$正则化项是非凸、非光滑的函数，因此 `LS-PLM` 的目标函数 采用传统的算法难以优化。