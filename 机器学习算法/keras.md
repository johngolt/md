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