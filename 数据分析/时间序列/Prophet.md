#### Prophet

##### 原理

$$
y(t) = g(t)+s(t)+h(t)+\epsilon_t
$$

模型将时间序列分成3个部分的叠加，其中$g(t)$表示增长函数，用来拟合非周期性变化的。$s(t)$用来表示周期性变化，比如说每周，每年，季节等，$h(t)$表示假期，节日等特殊原因等造成的变化，最后$\epsilon_t$为噪声项，用他来表示随机无法预测的波动，我们假设$\epsilon_t$是高斯的。

###### 增长项

我们将增长项$g(t)$定义为了一个逻辑函数：
$$
g(t) = \frac{C}{1+\exp(-k(t-b))}
$$
其中$C$是人口容量，$k$是增长率，$b$是偏移量。显然随着$t$增加，$g(t)$越趋于$C$，$k$越大，增长速度就越快。

下面假设已经放置了$S$个变点了，并且变点的位置是在时间戳$s_j,1\le j\le S$上，那么在这些时间戳上，我们就需要给出增长率的变化，也就是在时间戳$s_j$上发生的 change in rate。可以假设有这样一个向量：$\delta\in \mathbb{R}^S$，其中$\delta_j$表示在时间戳$s_j$上的增长率的变化量。如果一开始的增长率我们使用$k$来代替的话，那么在时间戳 $t$上的增长率就是$k+\sum_{j:j>s_j}\delta_j$，通过一个指示函数$\mathbf{a}(t)\in\{0,1\}^S$就是
$$
a_j(t)=\left\{\begin{array}{ll}{1,} & {t \ge s_j} \\ {0,} & { otherwise}\end{array}\right.
$$
那么在时间戳$t$上面的增长率就是$k+\mathbf{a}^T\delta$。一旦变化量$k$确定了，另外一个参数$b$也要随之确定。在这里需要把线段的边界处理好，因此通过数学计算可以得到：
$$
\gamma_j =(s_j-b-\sum_{l<j}\gamma_l)(1-\frac{k+\sum_{l<j}\delta_l}{k+\sum_{l\le j}\delta_l})
$$
于是最后我们可以得到分段的逻辑回归增长模型就是
$$
g(t)=\frac{C(t)}{1+\exp(-(k+\mathbf{a}(t)^T\mathbf{\delta})(t-(b-\mathbf{a}(t)^T\mathbf{\gamma})))}
$$
其中$\mathbf{a}(t) = (a_1(t),\cdots,a_S(t))^T, \mathbf{\delta}=(\delta_1,\cdots,\delta_S)^T, \gamma=(\gamma_1,\cdots,\gamma_S)^T$

另外，基于分段线性函数的模型形如：$g(t)=(k+\mathbf{a}(t)^T\mathbf{\delta})t+(b+\mathbf{a}(t)^T\mathbf{\gamma}))$

$k$是增长率, $\delta$表示增长率的变化量，$b$是偏移量参数，$\gamma=(\gamma_1,\cdots,\gamma_S)^T, \gamma_j=-s_j\delta_j$

###### 季节性

几乎所有的时间序列预测模型都会考虑这个因素，因为时间序列通常会随着天，周，月，年等季节性的变化而呈现季节性的变化，也称为周期性的变化。假设$P$表示时间序列的周期，$P=365.25$表示以年为周期，$P=7$表示以周为周期。它的傅里叶级数的形式都是：
$$
s(t)=\sum_{n=-N}^{N}c_ne^{i\frac{2\pi nt}{P}}=\sum_{n=1}^N(a_ncos(\frac{2\pi nt}{P})+b_nsin(\frac{2\pi nt}{P}))
$$
依照经验，对于一年为周期的序列而言，$N=10$；对于以周为周期的序列而言，$N=3$。这里的参数可以形成列向量：$\beta=(a_1,b_1,\cdots,a_N,b_N)$。

当$N=10$时
$$
X(t) = [\cos(\frac{2\pi (1)t}{365.25}),\cdots,\sin(\frac{2\pi (10)t}{365.25})]
$$
因此时间序列的季节项就是：$s(t) = X(t)\beta$，而$\beta$的初始化时$\beta\sim N(0,\sigma^2)$。这里的$\sigma$时通过超参数来控制的，也就是$\sigma$这个值越大，表示季节的效应越明显；这个值越小，表示季节的效应越不明显。

###### 节假日

对于第$i$个节假日来说，$D_i$表示该节假日的前后一段时间。为了表示节假日效应，我们需要一个相应的指示函数，同时需要一个参数$\mathcal{k}_i$来表示节假日的影响范围。假设我们由$L$个节假日，那么
$$
\begin{array}{l} h(t) = Z(t)\mathbf{k}=\sum_{i=1}^L\mathcal{k}_iI(t\in D_i)\\
Z(t) = [I(t\in D_1),\cdots, I(t \in D_L)]\\
\mathbf{k}=(k_1,\cdots,k_l)^T. 
\end{array}
$$
其中$\mathbf{k}\sim N(0,\nu^2)$并且该正态分布是受到$\nu=\text{prior scale}$这个指标影响的。默认值是 10，当值越大时，表示节假日对模型的影响越大；当值越小时，表示节假日对模型的效果越小。

##### 文档

###### Saturating Forecasts

By default, Prophet uses a linear model for its forecast. When forecasting growth, there is usually some maximum achievable point: total market size, total population size, etc. This is called the carrying capacity, and the forecast should saturate at this point. Prophet allows you to make forecasts using a logistic growth trend model, with a specified carrying capacity. We must specify the carrying capacity in a column `cap`. The important things to note are that `cap` must be specified for every row in the `dataframe`, and that it does not have to be constant. If the market size is growing, then `cap` can be an increasing sequence. We make a `dataframe` for future predictions as before, except we must also specify the capacity in the future. The logistic function has an implicit minimum of 0, and will saturate at 0 the same way that it saturates at the capacity. The logistic growth model can also handle a saturating minimum, which is specified with a column `floor` in the same way as the `cap` column specifies the maximum. To use a logistic growth trend with a saturating minimum, a maximum capacity must also be specified.

```python
df['cap'] = 6
df['floor'] = 1.5
m = Prophet(growth='logistic')
m.fit(df)
future = m.make_future_dataframe(periods=1826)
future['cap'] = 6
future['floor'] = 1.5
fcst = m.predict(future)
fig = m.plot(fcst)
```

###### Trend Changepoints

real time series frequently have abrupt changes in their trajectories. By default, Prophet will automatically detect these `changepoints` and will allow the trend to adapt appropriately. However, if you wish to have finer control over this process, then there are several input arguments you can use.

By default, Prophet specifies 25 potential changepoints which are uniformly placed in the first 80% of the time series. The vertical lines in this figure indicate where the potential changepoints were placed

The number of potential changepoints can be set using the argument `n_changepoints`, but this is better tuned by adjusting the regularization. The locations of the signification changepoints can be visualized with

```python
from fbprophet.plot import add_changepoints_to_plot
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)
```

If the trend changes are being overfit or underfit, you can adjust the strength of the sparse prior using the input argument `changepoint_prior_scale`. By default, this parameter is set to 0.05. Increasing it will make the trend *more* flexible

```python
m = Prophet(changepoint_prior_scale=0.5)
forecast = m.fit(df).predict(future)
fig = m.plot(forecast)
```

If you wish, rather than using automatic changepoint detection you can manually specify the locations of potential changepoints with the `changepoints` argument.

```python
m = Prophet(changepoints=['2014-01-01'])
forecast = m.fit(df).predict(future)
fig = m.plot(forecast)
```

###### Seasonality, Holiday Effects, And Regressor

**Modeling Holidays**：If you have holidays or other recurring events that you’d like to model, you must create a dataframe for them. It has two columns (`holiday` and `ds`) and a row for each occurrence of the holiday. It must include all occurrences of the holiday, both in the past (back as far as the historical data go) and in the future (out as far as the forecast is being made). If they won’t repeat in the future, Prophet will model them and then not include them in the forecast. You can also include columns `lower_window` and `upper_window` which extend the holiday out to `[lower_window, upper_window]` days around the date. 

```python
playoffs = pd.DataFrame({ 'holiday': 'playoff',
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                        '2010-01-24', '2010-02-07', '2011-01-08',
                        '2013-01-12', '2014-01-12', '2014-01-19',
                        '2014-02-02', '2015-01-11', '2016-01-17',
                        '2016-01-24', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,})
superbowls = pd.DataFrame({'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,})
holidays = pd.concat((playoffs, superbowls))
```

Above we have included the superbowl days as both playoff games and superbowl games. This means that the superbowl effect will be an additional additive bonus on top of the playoff effect.

Once the table is created, holiday effects are included in the forecast by passing them in with the `holidays` argument. 

```python
m = Prophet(holidays=holidays)
forecast = m.fit(df).predict(future)
```

You can use a built-in collection of country-specific holidays using the `add_country_holidays` method. he name of the country is specified, and then major holidays for that country will be included in addition to any holidays that are specified via the `holidays` argument described above

```python
m = Prophet(holidays=holidays)
m.add_country_holidays(country_name='US')
m.add_country_holidays(country_name='CN')
m.fit(df)
```

You can see which holidays were included by looking at the `train_holiday_names` attribute of the model.

**Seasonality**：The default values are often appropriate, but they can be increased when the seasonality needs to fit higher-frequency changes, and generally be less smooth. The Fourier order can be specified for each built-in seasonality when instantiating the model, here it is increased to 20:

```
from fbprophet.plot import plot_yearly
m = Prophet(yearly_seasonality=20).fit(df)
a = plot_yearly(m)
```

Increasing the number of Fourier terms allows the seasonality to fit faster changing cycles, but can also lead to overfitting: N Fourier terms corresponds to 2N variables used for modeling the cycle

Prophet will by default fit weekly and yearly seasonalities, if the time series is more than two cycles long. It will also fit daily seasonality for a sub-daily time series. You can add other seasonalities (monthly, quarterly, hourly) using the `add_seasonality` method

```python
m = Prophet(weekly_seasonality=False)
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
forecast = m.fit(df).predict(future)
fig = m.plot_components(forecast)
```

In some instances the seasonality may depend on other factors, such as a weekly seasonal pattern that is different during the summer than it is during the rest of the year, or a daily seasonal pattern that is different on weekends vs. on weekdays. These types of seasonalities can be modeled using conditional seasonalities.

```python
def is_nfl_season(ds):  
    date = pd.to_datetime(ds)    
    return (date.month > 8 or date.month < 2) 
df['on_season'] = df['ds'].apply(is_nfl_season) 
df['off_season'] = ~df['ds'].apply(is_nfl_season)

m = Prophet(weekly_seasonality=False)
m.add_seasonality(name='weekly_on_season', period=7, fourier_order=3, condition_name='on_season')
m.add_seasonality(name='weekly_off_season', period=7, fourier_order=3, condition_name='off_season')

future['on_season'] = future['ds'].apply(is_nfl_season)
future['off_season'] = ~future['ds'].apply(is_nfl_season)
forecast = m.fit(df).predict(future)
fig = m.plot_components(forecast)

m = Prophet()
m.add_seasonality(
    name='weekly', period=7, fourier_order=3, prior_scale=0.1)
```

**Regressors**：Additional regressors can be added to the linear part of the model using the `add_regressor` method or function. A column with the regressor value will need to be present in both the fitting and prediction dataframes. 

```python
def nfl_sunday(ds):
    date = pd.to_datetime(ds)
    if date.weekday() == 6 and (date.month > 8 or date.month < 2):
        return 1
    else:
        return 0
df['nfl_sunday'] = df['ds'].apply(nfl_sunday)

m = Prophet()
m.add_regressor('nfl_sunday')
m.fit(df)

future['nfl_sunday'] = future['ds'].apply(nfl_sunday)

forecast = m.predict(future)
fig = m.plot_components(forecast)
```

###### Multiplicative Seasonality

By default Prophet fits additive seasonalities, meaning the effect of the seasonality is added to the trend to get the forecast.Prophet can model multiplicative seasonality by setting `seasonality_mode='multiplicative'` in the input arguments

```python
m = Prophet(seasonality_mode='multiplicative')
m.fit(df)
forecast = m.predict(future)
fig = m.plot(forecast)
```

With `seasonality_mode='multiplicative'`, holiday effects will also be modeled as multiplicative. Any added seasonalities or extra regressors will by default use whatever `seasonality_mode` is set to, but can be overriden by specifying `mode='additive'` or `mode='multiplicative'` as an argument when adding the seasonality or regressor.

```python
m = Prophet(seasonality_mode='multiplicative')
m.add_seasonality('quarterly', period=91.25, fourier_order=8, mode='additive')
m.add_regressor('regressor', mode='additive')
```

Additive and multiplicative extra regressors will show up in separate panels on the components plot.

######  Uncertainty Intervals

By default Prophet will return uncertainty intervals for the forecast `yhat`. There are several important assumptions behind these uncertainty intervals. There are three sources of uncertainty in the forecast: uncertainty in the trend, uncertainty in the seasonality estimates, and additional observation noise.

we do the most reasonable thing we can, and we assume that the future will see similar trend changes as the history. In particular, we assume that the average frequency and magnitude of trend changes in the future will be the same as that which we observe in the history. We project these trend changes forward and by computing their distribution we obtain uncertainty intervals. One property of this way of measuring uncertainty is that allowing higher flexibility in the rate, by increasing `changepoint_prior_scale`, will increase the forecast uncertainty. This is because if we model more rate changes in the history then we will expect more in the future, and makes the uncertainty intervals a useful indicator of overfitting.

By default Prophet will only return uncertainty in the trend and observation noise. To get uncertainty in seasonality, you must do full Bayesian sampling. This is done using the parameter `mcmc.samples` (which defaults to 0).

###### Outliers

The best way to handle outliers is to remove them - Prophet has no problem with missing data. If you set their values to `NA` in the history but leave the dates in `future`, then Prophet will give you a prediction for their values.

```python
df.loc[(df['ds'] > '2010-01-01') & (df['ds'] < '2011-01-01'), 'y'] = None
model = Prophet().fit(df)
fig = model.plot(model.predict(future))
```

###### Non-Daily Data