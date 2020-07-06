```python
import matplotlib.animation as ani
animator = ani.FuncAnimation(fig, chartfunc, interval = 100)
```

| 名称        | 作用                                             |
| ----------- | ------------------------------------------------ |
| `fig`       | 用来 「绘制图表」的 figure 对象                  |
| `chartfunc` | 一个以数字为输入的函数，其含义为时间序列上的时间 |
| `interval`  | 帧之间的间隔延迟，以毫秒为单位，默认值为 200     |

