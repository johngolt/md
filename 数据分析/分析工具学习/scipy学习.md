### 常数和特殊函数

#### $\text{constants}$模块

`constants`模块还包含了许多单位信息，它们是1单位的量转换成标准单位时的数值：

#### $\text{special}$模块

`scipy`的`special`模块是个非常完整的函数库，其中包含了基本数学函数、特殊数学函数以及`numpy`中出现的所有函数。这些特殊函数都是`ufunc`函数，支持数组的广播运算。

### 拟合和优化

`scipy`的`optimize`模块提供了许多数值优化算法。

###### 求解非线性方程组

```python
 scipy.optimize.fsolve(func, x0, args=(), fprime=None, full_output=0, col_deriv=0,
  xtol=1.49012e-08, maxfev=0, band=None, epsfcn=None, factor=100, diag=None)
```

- `func`：是个可调用对象，它代表了非线性方程组。给他传入方程组的各个参数，它返回各个方程残差
- `x0`：预设的方程的根的初始值
- `args`：一个元组，用于给`func`提供额外的参数。
- `fprime`：用于计算`func`的雅可比矩阵按行排列。默认情况下，算法自动推算
- `full_output`：如果为`True`，则给出更详细的输出
- `col_deriv`：如果为`True`，则计算雅可比矩阵更快
- `diag`：它给出了每个变量的缩放因子

返回值：`x`：方程组的根组成的数组

假设待求解的方程组为：
$$
\begin{aligned} f_{1}\left(x_{1}, x_{2}, x_{3}\right) &=0 \\ f_{2}\left(x_{1}, x_{2}, x_{3}\right) &=0 \\ f_{3}\left(x_{1}, x_{2}, x_{3}\right) &=0 \end{aligned}
$$
那么我们的`func`函数为

```python
def func(x):
    x1, x2, x3=x.tolist()
    return np.array([f1(x1, x2, x3), f2(x1, x2, x3), f3(x1, x2, x3)])
```

而雅可比矩阵为：
$$
J=\left[\begin{array}{ccc}{\frac{\partial f_{1}}{\partial x_{1}}} & {\frac{\partial f_{1}}{\partial x_{2}}} & {\frac{\partial f_{1}}{\partial x_{3}}} \\ {\frac{\partial f_{2}}{\partial x_{1}}} & {\frac{\partial f_{2}}{\partial x_{2}}} & {\frac{\partial f_{2}}{\partial x_{3}}} \\ {\frac{\partial f_{3}}{\partial x_{1}}} & {\frac{\partial f_{3}}{\partial x_{2}}} & {\frac{\partial f_{3}}{\partial x_{3}}}\end{array}\right]
$$

```python
def fprime(x):
    x1, x2, x3=x.tolist()
    return np.array([[df1/dx1, df1/dx2, df1/dx3], 
                    [df2/dx1, df2/dx2, df2/dx3], [df3/dx1,df3/dx2, df3/dx3]])
```

###### 最小二乘法拟合数据

```python
  scipy.optimize.leastsq(func, x0, args=(), Dfun=None, full_output=0, col_deriv=0,
  ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0, maxfev=0, epsfcn=None, 
  factor=100, diag=None)
```

- `func`：是个可调用对象，给出每次拟合的残差。最开始的参数是待优化参数；后面的参数由`args`给出
- `x0`：初始迭代值
- `args`：一个元组，用于给`func`提供额外的参数。
- `Dfun`：用于计算`func`的雅可比矩阵。默认情况下，算法自动推算。它给出残差的梯度。最开始的参数是待优化参数；后面的参数由`args`给出

返回值：`x`：拟合解组成的数组

假设我们拟合的函数是$f(x, y;a, b,c)=0$，其中$a,b,c$为参数。假设数据点的横坐标为$X$，纵坐标为$Y$，那么我们可以给出`func`为：

```python
def func(p, x, y):
    a, b, c = p.tolist()
    return f(x, y;a, b, c)
```

而雅可比矩阵$\left[\frac{\partial f}{\partial a}, \frac{\partial f}{\partial b}, \frac{\partial f}{\partial c}\right]$为

```python
def fprime(p, x, y):
    a, b, c = p.tolist()
    return np.c_[df/da, df/db, df/dc]
```

```python
import numpy as np
from scipy.optimize import leastsq
x = np.linspace(-10, 10, num=100)
y = np.sin(x)+np.random.randint(-1,1, 100)
def func(p, x, y):
    a,b,c,d = p.tolist()
    return a*x**3+b*x**2+c*x+d-y
def jac(p, x, y):
    a, b, c, d = p.tolist()
    return np.c_[x**3, x**2, x, np.ones_like(x)]
result = leastsq(func, x0=np.array([1, 1, 1, 1]), args=(x, y), Dfun=jac)
```

`scipy`提供了另一个函数来执行最小二乘法的曲线拟合：

```
 scipy.optimize.curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, 
    check_finite=True, bounds=(-inf, inf), method=None, **kwargs)
```

- `f`：可调用函数，它的优化参数被直接传入。其第一个参数一定是`xdata`，后面的参数是待优化参数
- `xdata`：`x`坐标
- `ydata`：`y`坐标
- `p0`：初始迭代值
- `sigma`：`y`值的不确定性的度量

```python
result = curve_fit(func, x, y, p0=np.array([1, 1, 1, 1]))
```

###### 求函数最小值

```python
  scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, 
  hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
```

- `fun`：可调用对象，待优化的函数。最开始的参数是待优化的自变量；后面的参数由`args`给出
- `x0`：自变量的初始迭代值
- `args`：一个元组，提供给`fun`的额外的参数
- `method`：一个字符串，指定了最优化算法。
- `jac`：一个可调用对象，雅可比矩阵。如果`jac`是个布尔值且为`True`，则会假设`fun`会返回梯度；如果是个布尔值且为`False`，则雅可比矩阵会被自动推断。
- `hess/hessp`：可调用对象，海森矩阵。二者只需要给出一个就可以，如果给出了`hess`，则忽略`hessp`。如果二者都未提供，则海森矩阵自动推断
- `bounds`：一个元组的序列，给定了每个自变量的取值范围。如果某个方向不限，则指定为`None`。每个范围都是一个`(min,max)`元组。
- `constrants`：一个字典或者字典的序列，给出了约束条件。只在`COBYLA/SLSQP`中使用。字典的键为：`type`：给出了约束类型。如`'eq'`代表相等；`'ineq'`代表不等、`fun`：给出了约束函数、`jac`：给出了约束函数的雅可比矩阵、`args`：一个序列，给出了传递给`fun`和`jac`的额外的参数

返回值：返回一个`OptimizeResult`对象。其重要属性为：`x`：最优解向量、`success`：一个布尔值，表示是否优化成功、`message`：描述了迭代终止的原因

假设我们要求解最小值的函数为：$f(x, y)=(1-x)^{2}+100\left(y-x^{2}\right)^{2}$

```python
import numpy as np
from scipy.optimize import minimize
def func(p):
    x, y = p.tolist()
    return (1-x)**2+100*(y-x**2)**2
def jac(p):
    return np.array([2*(x-1)+400*x*(x**2-y), 200*(y-x**2)])
def hess(p):
    x, y = p.tolist()
    return np.array([[400*(2*x**2-y)+2, -400*x], [-400*x, 200]])
result = minimize(func, x0=np.aray([10, 10]), method='Newton-CG', jac=jac, hess=hess)
```

常规的最优化算法很容易陷入局部极值点。`basinhopping`算法是一个寻找全局最优点的算法。

```python
scipy.optimize.basinhopping(func, x0, niter=100, T=1.0, stepsize=0.5, 
  minimizer_kwargs=None,take_step=None, accept_test=None, callback=None, 
  interval=50, disp=False, niter_success=None)
```

- `func`：可调用函数。为待优化的目标函数。最开始的参数是待优化的自变量；后面的参数由`minimizer_kwargs`字典给出
- `x0`：一个向量，设定迭代的初始值
- `niter`：一个整数，指定迭代次数
- `take_step`：一个可调用对象，给出了游走策略
- `accept_step`：一个可调用对象，用于判断是否接受这一步
- `callback`：一个可调用对象，每当有一个极值点找到时，被调用
- `interval`：一个整数，指定`stepsize`被更新的频率

返回值：一个`OptimizeResult`对象。其重要属性为：`x`：最优解向量

```python
from scipy.optimize import basinhopping
result = basinhopping(func, x0=np.array([10, 10]))
```

### 线性代数

###### 求解线性方程组

在`numpy`中使用`numpy.linalg.solve(a, b)`。而`scipy`中的求解线性方程组：

```python
scipy.linalg.solve(a, b, sym_pos=False, lower=False, overwrite_a=False, 
  overwrite_b=False, debug=False, check_finite=True)
```

- `a`：方阵，形状为 `(M,M)`
- `b`：一维向量，形状为`(M,)`。它求解的是线性方程组 。如果有  个线性方程组要求解，且 `a`，相同，则 `b`的形状为 `(M,k)`
- `sym_pos`：一个布尔值，指定`a`是否正定的对称矩阵
- `lower`：一个布尔值。如果`sym_pos=True`时：如果为`lower=True`，则使用`a`的下三角矩阵。默认使用`a`的上三角矩阵。
- `check_finite`：如果为`True`，则检测输入中是否有`nan`或者`inf`

返回线性方程组的解。

###### 矩阵的`LU`分解：

矩阵`LU`分解：$\mathbf{A}=\mathbf{P} \mathbf{L} \mathbf{U}$。其中：$\mathbf{P}$为转置矩阵。$\mathbf{L}$为单位下三角矩阵对角线元素为1，$\mathbf{P}$为上三角矩阵对角线元素为0

```python
scipy.linalg.lu_factor(a, overwrite_a=False, check_finite=True)
```

- `a`：方阵，形状为`(M,M)`，要求非奇异矩阵；
- `overwrite_a`：一个布尔值，指定是否将结果写到`a`的存储区。

返回:`lu`：一个数组，形状为`(N,N)`，该矩阵的上三角矩阵就是`U`，下三角矩阵就是`L`、`piv`：一个数组，形状为`(N,)`。它给出了`P`矩阵：矩阵`a`的第 `i`行被交换到了第`piv[i]`行

当对矩阵进行了`LU`分解之后，可以方便的求解线性方程组。

```python
scipy.linalg.lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True)
```

- `lu_and_piv`：一个元组，由`lu_factor`返回
- `b`：一维向量，形状为`(M,)`。它求解的是线性方程组$\mathbf{A} \mathbf{x}=\mathbf{b}$。如果有$k$个线性方程组要求解，且 `a`，相同，则 `b`的形状为 `(M,k)`
- `trans`：指定求解类型: 如果为 0 ，则求解： $\mathbf{A} \mathbf{x}=\mathbf{b}$。如果为 1 ，则求解：$\mathbf{A}^T \mathbf{x}=\mathbf{b}$。如果为 2 ，则求解：$\mathbf{A}^H \mathbf{x}=\mathbf{b}$

`lstsq`比`solve`更一般化，它不要求矩阵$\mathbf{A}$是方阵。 它找到一组解$\mathbf{x}$，使得$\|\mathbf{b}-\mathbf{A} \mathbf{x}\|$最小，我们称得到的结果为最小二乘解。

```
scipy.linalg.lstsq(a, b, cond=None, overwrite_a=False, overwrite_b=False,check_finite=True, lapack_driver=None)
```

- `a`：为矩阵，形状为`(M,N)`
- `b`：一维向量，形状为`(M,)`。它求解的是线性方程组 。如果有  个线性方程组要求解，且 `a`，相同，则 `b`的形状为 `(M,k)`
- `cond`：一个浮点数，去掉最小的一些特征值。当特征值小于`cond * largest_singular_value`时，该特征值认为是零
- `lapack_driver`：一个字符串，指定求解算法。可以为：`'gelsd'/'gelsy'/'gelss'`。默认的`'gelsd'`效果就很好，但是在许多问题上`'gelsy'`效果更好。

###### 求解特征值和特征向量

```python
scipy.linalg.eig(a, b=None, left=False, right=True, overwrite_a=False, 
  overwrite_b=False, check_finite=True)
```

- `a`：一个方阵，形状为`(M,M)`。待求解特征值和特征向量的矩阵。
- `b`：默认为`None`，表示求解标准的特征值问题： 。 也可以是一个形状与`a`相同的方阵，此时表示广义特征值问题：  
- `left`：一个布尔值。如果为`True`，则计算左特征向量
- `right`：一个布尔值。如果为`True`，则计算右特征向量

返回值：`w`：一个一维数组，代表了`M`特特征值。

- `vl`：一个数组，形状为`(M,M)`，表示正则化的左特征向量每个特征向量占据一列，而不是一行。仅当`left=True`时返回
- `vr`：一个数组，形状为`(M,M)`，表示正则化的右特征向量每个特征向量占据一列，而不是一行。仅当`right=True`时返回

右特征值：$\mathbf{A} \mathbf{x}_{r}=\lambda \mathbf{x}_{r}$；左特征值：$\mathbf{A}^{H} \mathbf{x}_{l}=\operatorname{con} j(\lambda) \mathbf{x}_{l}$，其中$conj(\lambda)$为特征值的共轭。令$\mathbf{P}=\left[\mathbf{x}_{r 1}, \mathbf{x}_{r 2}, \cdots, \mathbf{x}_{r M}\right]$，令
$$
\mathbf{\Sigma}=\left[\begin{array}{ccccc}{\lambda_{1}} & {0} & {0} & {\cdots} & {0} \\ {0} & {\lambda_{2}} & {0} & {\cdots} & {0} \\ {\vdots} & {\vdots} & {\vdots} & {\ddots} & {\vdots} \\ {0} & {0} & {0} & {\cdots} & {\lambda_{M}}\end{array}\right]
$$
则有：$\mathbf{A} \mathbf{P}=\mathbf{P} \boldsymbol{\Sigma} \Longrightarrow \mathbf{A}=\mathbf{P} \boldsymbol{\Sigma} \mathbf{P}^{-1}$

###### 矩阵的奇异值分解

 设矩阵$\mathbf{A}$为$M\times N$阶的矩阵，则存在一个分解，使得：$\mathbf{A}=\mathbf{U} \Sigma \mathbf{V}^{H}$，其中$\mathbf{U}$为$M\times M$阶酉矩阵；$\boldsymbol{\Sigma}$为半正定的$M\times N$阶的对焦矩阵；而$\mathbf{V}$为$N\times N$阶酉矩阵。

 $\Sigma$对角线上的元素为$\mathbf{A}$的奇异值，通常按照从大到小排列。

```python
scipy.linalg.svd(a, full_matrices=True, compute_uv=True, overwrite_a=False, 
  check_finite=True, lapack_driver='gesdd')
```

- `a`：一个矩阵，形状为`(M,N)`，待分解的矩阵。
- `full_matrices`：如果为`True`，则$\mathbf{U}$的形状为`(M,M)`、$\mathbf{V}^H$的形状为`(N,N)`；否则$\mathbf{U}$的形状为`(M,K)`、$\mathbf{V}^H$的形状为`(K,N)`，其中 `K=min(M,N)`
- `compute_uv`：如果`True`，则结果中额外返回`U`以及`Vh`；否则只返回奇异值
- `lapack_driver`：一个字符串，指定求解算法。可以为：`'gesdd'/'gesvd'`。默认的`'gesdd'`。

返回值：`U`：$\mathbf{U}$矩阵。`s`：奇异值，它是一个一维数组，按照降序排列。长度为 `K=min(M,N)`、`Vh`：就是$\mathbf{V}^H$矩阵

### 统计

`scipy`中的`stats`模块中包含了很多概率分布的随机变量。所有的连续随机变量都是`rv_continuous`的派生类的对象。所有的离散随机变量都是`rv_discrete`的派生类的对象

#### 连续随机变量

连续随机变量对象都有如下方法：

| 函数                             | 作用                                                         |
| -------------------------------- | ------------------------------------------------------------ |
| `rvs(*args, **kwds)`             | 获取该分布的一个或者一组随机值                               |
| `pdf(x, *args, **kwds)`          | 概率密度函数在`x`处的取值                                    |
| `logpdf(x, *args, **kwds)`       | 概率密度函数在`x`处的对数值                                  |
| `cdf(x, *args, **kwds)`          | 累积分布函数在`x`处的取值                                    |
| `logcdf(x, *args, **kwds)`       | 累积分布函数在`x`处的对数值                                  |
| `sf(x, *args, **kwds)`           | 生存函数在`x`处的取值，它等于`1-cdf(x)`                      |
| `ppf(q, *args, **kwds)`          | 累积分布函数的反函数                                         |
| `moment(n, *args, **kwds)`       | n-th order non-central moment of distribution.               |
| `stats(*args, **kwds)`           | 计算随机变量的期望值和方差值等统计量                         |
| `fit(data, *args, **kwds)`       | 对一组随机取样进行拟合，找出最适合取样数据的概率密度函数的系数 |
| `nnlf(theta, x)`                 | 返回负的似然函数                                             |
| `interval(alpha, *args, **kwds)` | Confidence interval with equal areas around the median.      |

其中的`args/kwds`参数可能为：

- `arg1, arg2, arg3,...`: array_like. The shape parameter(s) for the distribution
- `loc` : array_like. location parameter
- `scale` : array_like. scale parameter
- `size` : int or tuple of ints. Defining number of random variates.
- `random_state` : None or int or `np.random.RandomState` instance。If int or `RandomState`, use it for drawing the random variates. If None, rely on `self.random_state`. Default is None.

#### 离散随机变量

离散随机变量对象都有如下方法：

| 函数                                     | 作用                      |
| ---------------------------------------- | ------------------------- |
| `rvs(<shape(s)>, loc=0, size=1)`         | 生成随机值                |
| `pmf(x, <shape(s)>, loc=0)`              | 概率密度函数在`x`处的值   |
| `cdf(x, <shape(s)>, loc=0)`              | 累积分布函数在`x`处的取值 |
| `sf(x, <shape(s)>, loc=0)`               | 生存函数在`x`处的值       |
| `ppf(q, <shape(s)>, loc=0)`              | 累积分布函数的反函数      |
| `isf(q, <shape(s)>, loc=0)`              | 生存函数的反函数          |
| `stats(<shape(s)>, loc=0, moments='mv')` | 计算期望方差等统计量      |

#### 核密度估计

正态核密度估计：

```python
class scipy.stats.gaussian_kde(dataset, bw_method=None)
```

参数：`dataset`：被估计的数据集。`bw_method`：用于设定带宽 。

属性：`dataset`：被估计的数据集、`d`：数据集的维度、`n`：数据点的个数、`factor`：带宽、`covariance`：数据集的相关矩阵

方法：`evaluate(points)`：估计样本点的概率密度、`__call__(points)`：估计样本点的概率密度、`pdf(x)`：估计样本的概率密度

### 数值积分

`scipy`的`integrate`模块提供了集中数值积分算法，其中包括对常微分方程组`ODE`的数值积分。

#### 积分

##### 数值积分函数

```python
scipy.integrate.quad(func, a, b, args=(), full_output=0, epsabs=1.49e-08, 
  epsrel=1.49e-08, limit=50, points=None, weight=None, wvar=None,
  wopts=None, maxp1=50, limlst=50)
```

- `func`：一个`Python`函数对象，代表被积分的函数。如果它带有多个参数，则积分只在第一个参数上进行。其他参数，则由`args`提供
- `a`：积分下限。用`-numpy.inf`代表负无穷
- `b`：积分上限。用`numpy.inf`代表正无穷
- `args`：额外传递的参数给`func`

返回值：`y`：一个浮点标量值，表示积分结果

计算曲线积分：$y=\sqrt{1-a x^{2}}$

```python
import numpy as np
import scipy.integrate as integrate
def func(x, a):
    return (1-a*x**2)**0.5
result = integrate.quad(-1, 1, args=(1,))
```

##### 二重定积分

```python
scipy.integrate.dblquad(func, a, b, gfun, hfun, args=(),
  epsabs=1.49e-08, epsrel=1.49e-08)
```

- `func`：一个`Python`函数对象，代表被积分的函数。它至少有两个参数：`y`和`x`。其中`y`为第一个参数，`x`为第二个参数。这两个参数为积分参数。如果有其他参数，则由`args`提供
- `a`：`x`的积分下限。用`-numpy.inf`代表负无穷
- `b`：`x`的积分上限。用`numpy.inf`代表正无穷
- `gfun`：`y`的下边界曲线。它是一个函数或者`lambda`表达式，参数为`x`,返回一个浮点数。
- `hfun`：`y`的上界曲线。参数为`x`,返回一个浮点数。

返回值：`y`：一个浮点标量值，表示积分结果

计算二重积分：$\int_{-1}^{1} \int_{-\sqrt{1-x^{2}}}^{\sqrt{1-x^{2}}} \sqrt{1-x^{2}-\frac{y^{2}}{2}} d y d x$

```python
def db1func(y, x):
    return (1-x**2-y**2/2)**0.5
def gfun(x):
    return -(1-x**2)**0.5
def hfun(x):
    return -gfun(x)
result = integrate.dblquad(func, -1, 1, gfun, hfun)
```

##### 三重定积分

```
scipy.integrate.tplquad(func, a, b, gfun, hfun, qfun, rfun, args=(), 
  epsabs=1.49e-08, epsrel=1.49e-08)
```

- `func`：一个`Python`函数对象，代表被积分的函数。它至少有三个参数：`z`、`y`和`x`。其中`z`为第一个参数，`y`为第二个参数，`x`为第三个参数。这三个参数为积分参数。如果有其他参数，则由`args`提供
- `a`：`x`的积分下限。用`-numpy.inf`代表负无穷
- `b`：`x`的积分上限。用`numpy.inf`代表正无穷
- `gfun`：`y`的下边界曲线。参数为`x`,返回一个浮点数。
- `hfun`：`y`的上界曲线。参数为`x`,返回一个浮点数。
- `qfun`：`z`的下边界曲面。第一个参数为`x`，第二个参数为`y`，返回一个浮点数。
- `rfun`：`z`的上边界曲面。第一个参数为`x`，第二个参数为`y`，返回一个浮点数。

返回值：`y`：一个浮点标量值，表示积分结果

#### 求解常微分方程组

```
scipy.integrate.odeint(func, y0, t, args=(), Dfun=None, col_deriv=0, full_output=0, 
  ml=None, mu=None, rtol=None, atol=None, tcrit=None, h0=0.0, hmax=0.0, 
  hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12, mxords=5, printmessg=0)
```

- `func`：梯度函数。第一个参数为`y`，第二个参数为`t0`，即计算`t0`时刻的梯度。其他的参数由`args`提供
- `y0`：初始的`y`
- `t`：一个时间点序列。
- `args`：额外提供给`func`的参数。
- `Dfun`：`func`的雅可比矩阵，行优先
- `col_deriv`：一个布尔值。如果`Dfun`未给出，则算法自动推导。该参数决定了自动推导的方式

返回值：`y`：一个数组，形状为 `(len(t),len(y0)`。它给出了每个时刻的`y`值

计算洛伦茨引子的轨迹
$$
\begin{array}{c}{\frac{d x}{d t}=\sigma(y-x)} \\ {\frac{d y}{d t}=x(\rho-z)-y} \\ {\frac{d z}{d t}=x y-\beta z}\end{array}
$$

```python
def func(w, t0, sigma, rho, beta):
    x, y, z = w.tolist()
    return np.array([sigma*(y-x), x*(rho-z)-y, x*y -beta*z])
def Dfunc(w, t0, sigma, rho, beta):
    x, y, z = w.tolist()
    return np.array([[-sigma, sigma, 0], [sigma-z, -1, -x], 
                    [y, x, -beta]])
t = np.linspace(0, 20, 5000)
t0=[0., 1., 0.]
t0_2=[0., 2, 0.]
track1 = integrate.odeint(func, t0, t, args=(10., 28., 3.), Dfunc=Dfunc)
```