#### stats

All distributions will have `location` and `Scale` parameters along with any shape parameters needed, the names for the shape parameters will vary. Standard form for the distributions will be given where `L=0.0` and `S=1.0`.

| Function Name                    | Standard Function                                         | Transformation                                               |
| -------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------ |
| Cumulative Distribution Function | $F(x)$                                                    | $F(x;L,S)=F(\frac{x-L}{S})$                                  |
| Probability Density Function     | $f(x)=F^{\prime}(x)$                                      | $f(x ; L, S)=\frac{1}{S} f\left(\frac{(x-L)}{S}\right)$      |
| Percent Point Function           | $G(q)=F^{-1}(q)$                                          | $G(q ; L, S)=L+S G(q)$                                       |
| Probability Sparsity Function    | $g(q)=G^{\prime}(q)$                                      | $g(q ; L, S)=S g(q)$                                         |
| Hazard Function                  | $h_{a}(x)=\frac{f(x)}{1-F(x)}$                            | $h_{a}(x ; L, S)=\frac{1}{S} h_{a}\left(\frac{(x-L)}{S}\right)$ |
| Cumulative Hazard Function       | $H_{a}(x)=\log \frac{1}{1-F(x)}$                          | $H_{a}(x ; L, S)=H_{a}\left(\frac{(x-L)}{S}\right)$          |
| Survival Function                | $S(x)=1-F(x)$                                             | $S(x ; L, S)=S\left(\frac{(x-L)}{S}\right)$                  |
| Inverse Survival Function        | $Z(\alpha)=S^{-1}(\alpha)=G(1-\alpha)$                    | $Z(\alpha ; L, S)=L+S Z(\alpha)$                             |
| Moment Generating Function       | $M_{Y}(t)=E\left[e^{Y t}\right]$                          | $M_{X}(t)=e^{L t} M_{Y}(S t)$                                |
| Entropy                          | $h[Y]=-\int f(y) \log f(y) d y$                           | $h[X]=h[Y]+\log S$                                           |
| Moments                          | $\mu_{n}^{\prime}=E\left[Y^{n}\right]$                    | $E\left[X^{n}\right]=L^{n} \sum_{k=0}^{N}\left(\begin{array}{l}{n} \\ {k}\end{array}\right)\left(\frac{S}{L}\right)^{k} \mu_{k}^{\prime}$ |
| Central Moment                   | $\mu_{n}=E\left[(Y-\mu)^{n}\right]$                       | $E\left[\left(X-\mu_{X}\right)^{n}\right]=S^{n} \mu_{n}$     |
| mean,var                         | $\mu,\mu_2$                                               | $L+S\mu, S^2\mu_2$                                           |
| skewness                         | $\gamma_{1}=\frac{\mu_{3}}{\left(\mu_{2}\right)^{3 / 2}}$ | $\gamma_{1}$                                                 |
| kurtosis                         | $\gamma_{2}=\frac{\mu_{4}}{\left(\mu_{2}\right)^{2}}-3$   | $\gamma_{2}$                                                 |

##### Summary Statistics

| Function Name                                  | Usage                                                        |
| ---------------------------------------------- | ------------------------------------------------------------ |
| `describe(a[,axis,ddof,bias])`                 | Compute several descriptive statistics of the passed array.  |
| `gmean(a[, axis, dtype])`                      | Compute the geometric mean along the specified axis.         |
| `hmean(a[, axis, dtype])`                      | Calculate the harmonic mean along the specified axis.        |
| `kurtosis(a[, axis])`                          |                                                              |
| `mode(a[, axis])`                              |                                                              |
| `variation(a[, axis])`                         | Compute the coefficient of variation, the ratio of the biased standard deviation to the mean. |
| `find_repeats(arr)`                            | Find repeats and repeat counts.                              |
| `iqr(x[, axis, rng, scale])`                   | Compute the interquartile range of the data along the specified axis. |
| `entropy(pk[,qk, base])`                       | Calculate the entropy of a distribution for given probability values. |
| `median_absolute_deviation(x[, axis, center])` | Compute the median absolute deviation of the data along the given axis. |

```python
from scipy import stats
a = np.arange(10)
stats.describe(a)
stats.find_repeats([2, 1, 2, 3, 2, 2, 5])
#RepeatedResults(values=array([2.]), counts=array([4]))
```

The interquartile range is the difference between the 75th and 25th percentile of the data. When comparing the behavior of `median_absolute_deviation` with `np.std`, the latter is affected when we change a single value of an array to have an `outlier` value while the MAD hardly changes

##### Frequency Statistics

| Function Name                               | Usage                                                        |
| ------------------------------------------- | ------------------------------------------------------------ |
| `cumfreq(a[, numbins])`                     | Return a cumulative frequency histogram, using the histogram function. |
| `itemfreq(*args, **kwds)`                   |                                                              |
| `percentileofscore(a,per[, limit])`         | The percentile rank of a score relative to a list of scores. |
| `relfreq(a[, numbins])`                     | Return a relative frequency histogram, using the histogram function. |
| `binned_statistic(x, values[, statistics])` | Compute a binned statistic for one or more sets of data.     |

##### Correlation Functions

| Function Name                        | Usage                                                        |
| ------------------------------------ | ------------------------------------------------------------ |
| `f_oneway(*args)`                    | The one-way ANOVA tests the null hypothesis that two or more groups have the same population mean.  The test is applied to samples from two or more groups, possibly with differing sizes. |
| `pearsonr(x, y)`                     | The Pearson correlation coefficient measures the linear relationship<br/>between two datasets.  The calculation of the p-value relies on the<br/>assumption that each dataset is normally distributed. |
| `spearmanr(a[, b, axis])`            | The Spearman correlation is a nonparametric measure of the monotonicity<br/>of the relationship between two datasets. Unlike the Pearson correlation,<br/>the Spearman correlation does not assume that both datasets are normally<br/>distributed. |
| `pointbiserialr(x, y)`               | The point biserial correlation is used to measure the relationship<br/>between a binary variable, x, and a continuous variable, y. Like other<br/>correlation coefficients, this one varies between -1 and +1 with 0<br/>implying no correlation. Correlations of -1 or +1 imply a determinative<br/>relationship. |
| `kendalltau(x, y)`                   | Kendall's tau is a measure of the correspondence <br/>between two rankings. |
| `weightedtau(x, y[, rank, weigher])` | Compute a weighted version of `Kendall’s` τ.                 |
| `linregress(x[, y])`                 | Calculate a linear least-squares regression for two sets of measurements. |
| `siegelslopes(y[, x, method])`       | Computes the Siegel estimator for a set of points (x, y).    |
| `theilslopes(y[, x, alpha])`         | Computes the `Theil-Sen` estimator for a set of points (x, y). |

The `ANOVA` test has important assumptions that must be satisfied in order for the associated p-value to be valid.

1. The samples are independent.
2. Each sample is from a normally distributed population.
3. The population standard deviations of the groups are all equal. This property is known as `homoscedasticity`.

The point biserial correlation is used to measure the relationship between a binary variable, x, and a continuous variable, y. The value of the point-biserial correlation can be calculated from:
$$
r_{p b}=\frac{\overline{Y_{1}}-\overline{Y_{0}}}{s_{y}} \sqrt{\frac{N_{1} N_{2}}{N(N-1))}}
$$
`siegelslopes` implements a method for robust linear regression using repeated medians fit a line to the points (x, y). The method is robust to outliers with an asymptotic breakdown point of 50%.

##### Statistical Tests

| Function Name                                    | Usage                                                        |
| ------------------------------------------------ | ------------------------------------------------------------ |
| `ttest_1samp(a, popmean[, axis])`                | This is a two-sided test for the null hypothesis that the expected value<br/>(mean) of a sample of independent observations `a` is equal to the given<br/>population mean, `popmean`. |
| `ttest_ind(a, b[, axis, equal_var])`             | This is a two-sided test for the null hypothesis that 2 independent <br/>samples have identical average (expected) values. This test assumes that the<br/>populations have identical variances by default. |
| `ttest_ind_from_stats(mean1, std1, nobs1)`       | T-test for means of two independent samples from descriptive statistics. |
| `ttest_rel(a, b[, axis])`                        | This is a two-sided test for the null hypothesis that 2 related or repeated samples have identical average (expected) values. |
| `kstest(rvs, cdf[, args, N, alternative, mode])` | This performs a test of the distribution F(x) of an observed random variable against a given distribution G(x). Under the null hypothesis the two distributions are identical, F(x)=G(x). |
| `chisquare(f_obs[, f_exp, ddof, axis])`          | The chi square test tests the null hypothesis that the categorical data has the given frequencies. |
| `power_divergence(f_obs[, f_exp, ddof, axis])`   | This function tests the null hypothesis that the categorical data has the given frequencies, using the Cressie-Read power divergence statistic. |
| `ks_2samp(data1, data2[, alternative, mode])`    | Compute the `Kolmogorov-Smirnov` statistic on 2 samples.     |
| `epps_singleton_2samp(x, y[, t])`                | Test the null hypothesis that two samples have the same underlying probability distribution. |
| `mannwhitneyu(x, y[, use_continuity])`           | Compute the Mann-Whitney rank test on samples x and y.       |
| `tiecorrect(rankvals)`                           | Tie correction factor for ties in the Mann-Whitney U and<br/>Kruskal-Wallis H tests. |
| `rankdata(a[, method])`                          | Assign ranks to data, dealing with ties appropriately.       |
| `ranksums(x, y)`                                 | The Wilcoxon rank-sum test tests the null hypothesis that two sets<br/>of measurements are drawn from the same distribution. |
| `wilcoxon(x[, y, zero_method])`                  | The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from the same distribution. In particular, it tests whether the distribution of the differences x - y is symmetric about zero. |
| `kruskal(*args, **kwargs)`                       | The Kruskal-Wallis H-test tests the null hypothesis that the population<br/>median of all of the groups are equal. |
| `friedmanchiquare(*args)`                        | Compute the Friedman test for repeated measurements          |
| `combine_pvalues(pvalues[, method, weights])`    | Methods for combining the p-values of independent tests bearing upon the same hypothesis. |
| `jarque_bera(x)`                                 | The Jarque-Bera test tests whether the sample data has the skewness and<br/>kurtosis matching a normal distribution. |
| `ansari(x, y)`                                   | The Ansari-Bradley test is a non-parametric test for the equality<br/>of the scale parameter of the distributions from which two<br/>samples were drawn. |
| `bartlett(*args)`                                | Bartlett's test tests the null hypothesis that all input samples<br/>are from populations with equal variances. |
| `levene(*args, **kwds)`                          | The Levene test tests the null hypothesis that all input samples<br/>are from populations with equal variances. |
| `shapiro(x)`                                     | The Shapiro-Wilk test tests the null hypothesis that the<br/>data was drawn from a normal distribution. |
| `anderson(x[, dist])`                            | The Anderson-Darling tests the null hypothesis that a sample is<br/>drawn from a population that follows a particular <br/>distribution. |
| `binon_test(x[, n, p, alternative])`             | This is an exact, two-sided test of the null hypothesis<br/>that the probability of success in a Bernoulli experiment<br/>is `p`. |
| `fligner(*args, **kwds)`                         | Fligner's test tests the null hypothesis that all input samples<br/>are from populations with equal variances. |
| `median_test(*args, **kwds)`                     | Test that two or more samples come from population<br/>s with the same median. |
| `mood(x, y[, axis])`                             | Mood's two-sample test for scale parameters is a non-parametric test for the null hypothesis that two samples are drawn from the same distribution with the same scale parameter. |
| `skewtest\kurtosistest\normaltest(a[, axis])`    | Test whether the skew is different from the normal distribution. |
| `anderson_ksam(samples[, midrank])`              | It tests the null hypothesis<br/>that k-samples are drawn from the same population without having<br/>to specify the distribution function of that population. |

##### Transformations

| Function Name                                | Usage                                                        |
| -------------------------------------------- | ------------------------------------------------------------ |
| `boxcox(x[, lmbda, alpha])`                  | Return a positive dataset transformed by a Box-Cox power transformation. |
| `boxcox_normmax(x[, brack, method])`         | Compute optimal Box-Cox transform parameter for input data.  |
| `boxcox_llf(lmb, data)`                      | The boxcox log-likelihood function.                          |
| `yeojohnson(x[, lmbda])`                     | Return a dataset transformed by a Yeo-Johnson power transformation. |
| `obrientransform(*args)`                     | Compute the O’Brien transform on input data                  |
| `sigmaclip(a[, low, high])`                  | Iterative sigma-clipping of array elements.                  |
| `trimboth(a, proportiontocut[, tail, axis])` | Slices off a proportion of items from both ends of an array. |
| `trim1(a, proportiontocut[, tail, axis])`    | Slices off a proportion from ONE end of the passed array distribution. |
| `zmap(scores, compare[, axis, ddof])`        | Calculate the relative z-scores                              |
| `zscore(a[, axis, ddof])`                    | Calculate the z score of each value in the sample, relative to the sample mean and standard deviation. |

##### Statistical distance

| Function Name                              | Usage                                                        |
| ------------------------------------------ | ------------------------------------------------------------ |
| `wasserstein_distance(u_values, v_values)` | Compute the first `Wasserstein` distance between two `1D` distributions. |
| `energy_distance(u_values, v_values)`      | Compute the energy distance between two `1D` distributions.  |

##### Plot Tests

| Function Name                               | Usage                                                        |
| ------------------------------------------- | ------------------------------------------------------------ |
| `ppcc_max(x[, brack, dist])`                | Calculate the shape parameter that maximizes the PPCC        |
| `ppcc_plot(x, a, b[, dist, plot, N])`       | Calculate and optionally plot probability plot correlation coefficient. |
| `probplot(x, la, lb[, plot, N])`            | Calculate quantiles for a probability plot, and optionally show the plot. |
| `boxcox_normplot(x, la, lb[, plot, N])`     | Compute parameters for a Box-Cox normality plot, optionally show it. |
| `yeojohnson_normplot(x, la, bl[, plot, N])` | Compute parameters for a Yeo-Johnson normality plot, optionally show it. |

##### Univariate and Multivariate kernel density estimation

#### Fourier Transforms(`fftpack`)

##### Fast Fourier transforms

Fourier analysis is a method for expressing a function as a sum of periodic components, and for recovering the signal from those components. When both the function and its Fourier transform are replaced with discretized counterparts, it is called the discrete Fourier transform. 

The`FFT y[k]` of length `N` of the length-N sequence `x[n]` is defined as
$$
y[k]=\sum_{n=0}^{N-1} e^{-2 \pi j \frac{k n}{N}} x[n]
$$
and the inverse transform is defined as follows
$$
x[n]=\frac{1}{N} \sum_{k=0}^{N-1} e^{2 \pi j \frac{k n}{N}} y[k]
$$
The `FFT` input signal is inherently truncated. This truncation can be modeled as multiplication of an infinite signal with a rectangular window function. In the spectral domain this multiplication becomes convolution of the signal spectrum with the window function spectrum, being of form $\frac{sin⁡(x)}{x}$. This convolution is the cause of an effect called spectral leakage.  Windowing the signal with a dedicated window function helps mitigate spectral leakage.

The function `rfft` calculates the `FFT` of a real sequence and outputs the `FFT` coefficients `y[n]` with separate real and imaginary parts.

```python
from scipy.fftpack import fft
import matplotlib.pyplot as plt
N = 600
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
```

`fftpack.convolve`performs a convolution of two one-dimensional arrays in frequency domain.

##### Discrete Cosine Transforms

###### Type I `DCT`

$$
y[k]=x_{0}+(-1)^{k} x_{N-1}+2 \sum_{n=1}^{N-2} x[n] \cos \left(\frac{\pi n k}{N-1}\right), \quad 0 \leq k<N
$$

###### Type II `DCT`

$$
y[k]=2 \sum_{n=0}^{N-1} x[n] \cos \left(\frac{\pi(2 n+1) k}{2 N}\right) \quad 0 \leq k<N
$$

###### Type III `DCT`

$$
y[k]=x_{0}+2 \sum_{n=1}^{N-1} x[n] \cos \left(\frac{\pi n(2 k+1)}{2 N}\right) \quad 0 \leq k<N
$$