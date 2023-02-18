TFRecord 格式是一种用于存储二进制记录序列的简单格式。

`tf.Example` 消息（或 protobuf）是一种灵活的消息类型，表示 `{"string": value}` 映射。

```
import tensorflow as tf

import numpy as np
import IPython.display as display
```

##### `tf.Example`

从根本上讲，`tf.Example` 是 `{"string": tf.train.Feature}` 映射。

`tf.train.Feature`消息类型可以接受以下三种类型

大多数其他通用类型也可以强制转换成下面的其中一种：

tf.train.BytesList（可强制转换自以下类型）
string
byte
tf.train.FloatList（可强制转换自以下类型）
float (float32)
double (float64)
tf.train.Int64List（可强制转换自以下类型）
bool
enum
int32
uint32
int64
uint64

为了将标准 TensorFlow 类型转换为兼容 tf.Example 的 tf.train.Feature，可以使用下面的快捷函数。请注意，每个函数会接受标量输入值并返回包含上述三种 list 类型之一的 tf.train.Feature：

```python
# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
```

为了简单起见，本示例仅使用标量输入。要处理非标量特征，最简单的方法是使用 tf.io.serialize_tensor 将张量转换为二进制字符串。在 TensorFlow 中，字符串是标量。使用 tf.io.parse_tensor 可将二进制字符串转换回张量。

```
print(_bytes_feature(b'test_string'))

'''
bytes_list {
  value: "test_string"
}
'''
feature = _float_feature(np.exp(1))
feature.SerializeToString()
'''
b'\x12\x06\n\x04T\xf8-@'
'''
```

在每个观测结果中，需要使用上述其中一种函数，将每个值转换为包含三种兼容类型之一的 tf.train.Feature。

创建一个从特征名称字符串到第 1 步中生成的编码特征值的映射（字典）。

将第 2 步中生成的映射转换为 Features 消息。

```
# The number of observations in the dataset.
n_observations = int(1e4)

# Boolean feature, encoded as False or True.
feature0 = np.random.choice([False, True], n_observations)

# Integer feature, random from 0 to 4.
feature1 = np.random.randint(0, 5, n_observations)

# String feature
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]

# Float feature, from a standard normal distribution
feature3 = np.random.randn(n_observations)
```

```python
def serialize_example(feature0, feature1, feature2, feature3):
  """
  Creates a tf.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
  feature = {
      'feature0': _int64_feature(feature0),
      'feature1': _int64_feature(feature1),
      'feature2': _bytes_feature(feature2),
      'feature3': _float_feature(feature3),
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

serialized_example = serialize_example(False, 4, b'goat', 0.9876)
example_proto = tf.train.Example.FromString(serialized_example)
```

TFRecord 文件包含一系列记录。该文件只能按顺序读取。

不需要在 TFRecord 文件中使用 tf.Example。tf.Example 只是将字典序列化为字节字符串的一种方法。文本行、编码的图像数据，或序列化的张量（使用 tf.io.serialize_tensor，或在加载时使用 tf.io.parse_tensor）。

##### 写入`TFRecord`文件

要将数据放入数据集中，最简单的方式是使用 `from_tensor_slices` 方法。

```
features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
```

使用 tf.data.Dataset.map 方法可将函数应用于 Dataset 的每个元素。

映射函数必须在 TensorFlow 计算图模式下进行运算（它必须在 tf.Tensors 上运算并返回）。可以使用 tf.py_function 包装非张量函数（如 serialize_example）以使其兼容。

```python
def tf_serialize_example(f0,f1,f2,f3):
  tf_string = tf.py_function(
    serialize_example,
    (f0,f1,f2,f3),  # pass these args to the above function.
    tf.string)      # the return type is `tf.string`.
  return tf.reshape(tf_string, ()) # The result is a scalar
```

将此函数应用于数据集中的每个元素：

```
serialized_features_dataset = features_dataset.map(tf_serialize_example)
serialized_features_dataset
```

```
def generator():
  for features in features_dataset:
    yield serialize_example(*features)
    
serialized_features_dataset = tf.data.Dataset.from_generator(
    generator, output_types=tf.string, output_shapes=())
    
filename = 'test.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)
```

##### 读取`TFRecord`文件

```
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset
```

此时，数据集包含序列化的 tf.train.Example 消息。迭代时，它会将其作为标量字符串张量返回。

可以使用以下函数对这些张量进行解析。请注意，这里的 `feature_description` 是必需的，因为数据集使用计算图执行，并且需要以下描述来构建它们的形状和类型签名：

```python
# Create a description of the features.
feature_description = {
    'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
}

def _parse_function(example_proto):
  # Parse the input `tf.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)
```

或者，使用 tf.parse example 一次解析整个批次。使用 tf.data.Dataset.map 方法将此函数应用于数据集中的每一项

```
parsed_dataset = raw_dataset.map(_parse_function)
parsed_dataset
```

##### python中的`TFRecord`文件

```python
# Write the `tf.Example` observations to the file.
with tf.io.TFRecordWriter(filename) as writer:
  for i in range(n_observations):
    example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
    writer.write(example)
```

```
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
for raw_record in raw_dataset.take(1):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
```

