##### Time series

pandas captures 4 general time related concepts: Date times: A specific date and time with timezone support; Time deltas: An absolute time duration; Time spans: A span of time defined by a point in time and its associated frequency; Date offsets: A relative time duration that respects calendar arithmetic.

| 名称                     | 描述                   | 元素类型     | 创建方式                        |
| ------------------------ | ---------------------- | ------------ | ------------------------------- |
| Date Times(时间点/时刻)  | 描述特定日期或时间点   | `Timestamp`  | `to_datetime, date_range`       |
| Time Spans(时间段/时期)  | 由时间点定义的一段时期 | `Period`     | `Period, period_range`          |
| Date offsets(相对时间差) | 一段时间的相对大小     | `DateOffset` | `DateOffset`                    |
| Time deltas(绝对时间差)  | 一段时间的绝对大小     | `Timedelta`  | `to_timedelta,time_delta_range` |

事实上，Timestamp的精度远远不止day，可以最小到纳秒ns

```
pd.Timestamp.min#可用最小时间
pd.Timestamp.max
```

Timedelta绝对时间差的特点指无论是冬令时还是夏令时，增减1day都只计算24小时

DataOffset相对时间差指，无论一天是23\24\25小时，增减1day都与当天相同的时间保持一致

The power of ``datetime`` and ``dateutil`` lie in their flexibility and easy syntax: you can use these objects and their built-in methods to easily perform nearly any operation you might be interested in.
Where they break down is when you wish to work with large arrays of dates and times:
just as lists of Python numerical variables are suboptimal compared to NumPy-style typed numerical arrays, lists of Python datetime objects are suboptimal compared to typed arrays of encoded dates.

The weaknesses of Python's datetime format inspired the NumPy team to add a set of native time series data type to NumPy.
The ``datetime64`` dtype encodes dates as 64-bit integers, and thus allows arrays of dates to be represented very compactly.
The ``datetime64`` requires a very specific input format:

One detail of the ``datetime64`` and ``timedelta64`` objects is that they are built on a *fundamental time unit*.
Because the ``datetime64`` object is limited to 64-bit precision, the range of encodable times is $2^{64}$ times this fundamental unit.
In other words, ``datetime64`` imposes a trade-off between *time resolution* and *maximum time span*.

Notice that the time zone is automatically set to the local time on the computer executing the code.
You can force any desired fundamental unit using one of many format codes;

The following table,lists the available format codes
along with the relative and absolute timespans that they can encode:

| Code   | Meaning     | Time span (relative) | Time span (absolute)   |
| ------ | ----------- | -------------------- | ---------------------- |
| ``Y``  | Year        | ± 9.2e18 years       | [9.2e18 BC, 9.2e18 AD] |
| ``M``  | Month       | ± 7.6e17 years       | [7.6e17 BC, 7.6e17 AD] |
| ``W``  | Week        | ± 1.7e17 years       | [1.7e17 BC, 1.7e17 AD] |
| ``D``  | Day         | ± 2.5e16 years       | [2.5e16 BC, 2.5e16 AD] |
| ``h``  | Hour        | ± 1.0e15 years       | [1.0e15 BC, 1.0e15 AD] |
| ``m``  | Minute      | ± 1.7e13 years       | [1.7e13 BC, 1.7e13 AD] |
| ``s``  | Second      | ± 2.9e12 years       | [ 2.9e9 BC, 2.9e9 AD]  |
| ``ms`` | Millisecond | ± 2.9e9 years        | [ 2.9e6 BC, 2.9e6 AD]  |
| ``us`` | Microsecond | ± 2.9e6 years        | [290301 BC, 294241 AD] |
| ``ns`` | Nanosecond  | ± 292 years          | [ 1678 AD, 2262 AD]    |
| ``ps`` | Picosecond  | ± 106 days           | [ 1969 AD, 1970 AD]    |
| ``fs`` | Femtosecond | ± 2.6 hours          | [ 1969 AD, 1970 AD]    |
| ``as`` | Attosecond  | ± 9.2 seconds        | [ 1969 AD, 1970 AD]    |

Pandas builds upon all the tools just discussed to provide a ``Timestamp`` object, which combines the ease-of-use of ``datetime`` and ``dateutil`` with the efficient storage and vectorized interface of ``numpy.datetime64``.
From a group of these ``Timestamp`` objects, Pandas can construct a ``DatetimeIndex`` that can be used to index data in a ``Series`` or ``DataFrame``; 

any of the ``Series`` indexing patterns we discussed in previous sections, passing values that can be coerced into dates:

the fundamental Pandas data structures for working with time series data:

- For *time stamps*, Pandas provides the ``Timestamp`` type. As mentioned before, it is essentially a replacement for Python's native ``datetime``, but is based on the more efficient ``numpy.datetime64`` data type. The associated Index structure is ``DatetimeIndex``.
- For *time Periods*, Pandas provides the ``Period`` type. This encodes a fixed-frequency interval based on ``numpy.datetime64``. The associated index structure is ``PeriodIndex``.
- For *time deltas* or *durations*, Pandas provides the ``Timedelta`` type. ``Timedelta`` is a more efficient replacement for Python's native ``datetime.timedelta`` type, and is based on ``numpy.timedelta64``. The associated index structure is ``TimedeltaIndex``.

Any ``DatetimeIndex`` can be converted to a ``PeriodIndex`` with the ``to_period()`` function with the addition of a frequency code; 

A ``TimedeltaIndex`` is created,when a date is subtracted from another:

To make the creation of regular date sequences more convenient, Pandas offers a few functions for this purpose: ``pd.date_range()`` for timestamps, ``pd.period_range()`` for periods, and ``pd.timedelta_range()`` for time deltas.
We've seen that Python's ``range()`` and NumPy's ``np.arange()`` turn a startpoint, endpoint, and optional stepsize into a sequence.
Similarly, ``pd.date_range()`` accepts a start date, an end date, and an optional frequency code to create a regular sequence of dates.
By default, the frequency is one day:

Fundamental to these Pandas time series tools is the concept of a frequency or date offset.
The following table summarizes the main codes available:

| Code  | Description  | Code   | Description          |
| ----- | ------------ | ------ | -------------------- |
| ``D`` | Calendar day | ``B``  | Business day         |
| ``W`` | Weekly       |        |                      |
| ``M`` | Month end    | ``BM`` | Business month end   |
| ``Q`` | Quarter end  | ``BQ`` | Business quarter end |
| ``A`` | Year end     | ``BA`` | Business year end    |
| ``H`` | Hours        | ``BH`` | Business hours       |
| ``T`` | Minutes      |        |                      |
| ``S`` | Seconds      |        |                      |
| ``L`` | Milliseonds  |        |                      |
| ``U`` | Microseconds |        |                      |
| ``N`` | nanoseconds  |        |                      |

The monthly, quarterly, and annual frequencies are all marked at the end of the specified period.
By adding an ``S`` suffix to any of these, they instead will be marked at the beginning:

| Code   | Description   |      | Code    | Description            |
| ------ | ------------- | ---- | ------- | ---------------------- |
| ``MS`` | Month start   |      | ``BMS`` | Business month start   |
| ``QS`` | Quarter start |      | ``BQS`` | Business quarter start |
| ``AS`` | Year start    |      | ``BAS`` | Business year start    |

Additionally, you can change the month used to mark any quarterly or annual code by adding a three-letter month code as a suffix:

- ``Q-JAN``, ``BQ-FEB``, ``QS-MAR``, ``BQS-APR``, etc.
- ``A-JAN``, ``BA-FEB``, ``AS-MAR``, ``BAS-APR``, etc.

In the same way, the split-point of the weekly frequency can be modified by adding a three-letter weekday code:

- ``W-SUN``, ``W-MON``, ``W-TUE``, ``W-WED``, etc.

On top of this, codes can be combined with numbers to specify other frequencies.
For example, for a frequency of 2 hours 30 minutes, we can combine the hour (``H``) and minute (``T``) codes as follows:

All of these short codes refer to specific instances of Pandas time series offsets, which can be found in the ``pd.tseries.offsets`` module.

One common need for time series data is resampling at a higher or lower frequency.
This can be done using the ``resample()`` method, or the much simpler ``asfreq()`` method.
The primary difference between the two is that ``resample()`` is fundamentally a *data aggregation*, while ``asfreq()`` is fundamentally a *data selection*.

For up-sampling, ``resample()`` and ``asfreq()`` are largely equivalent, though resample has many more options available.
In this case, the default for both methods is to leave the up-sampled points empty, that is, filled with NA values.
Just as with the ``pd.fillna()`` function discussed previously, ``asfreq()`` accepts a ``method`` argument to specify how values are imputed.

Another common time series-specific operation is shifting of data in time.
Pandas has two closely related methods for computing this: ``shift()`` and ``tshift()``
In short, the difference between them is that ``shift()`` *shifts the data*, while ``tshift()`` *shifts the index*.
In both cases, the shift is specified in multiples of the frequency.