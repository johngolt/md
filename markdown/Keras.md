### Getting Started

#### Guide to the Sequential model

The model needs to know what input shape it should expect. For this reason, the first layer in a `Sequential` model needs to receive information about its input shape. There are several possible ways to do this: Pass an `input_shape` argument to the first layer. This is a shape tuple (a tuple of integers or `None` entries, where `None` indicates that any positive integer may be expected). In `input_shape`, the batch dimension is not included; Some `2D` layers, such as `Dense`, support the specification of their input shape via the argument `input_dim`, and some `3D` temporal layers support the arguments `input_dim`and `input_length`; If you ever need to specify a fixed batch size for your inputs, you can pass a `batch_size` argument to a layer. If you pass both `batch_size=32` and `input_shape=(6, 8)` to a layer, it will then expect every batch of inputs to have the batch shape `(32, 6, 8)`.

Before training a model, you need to configure the learning process, which is done via the `compile` method. It receives three arguments: An optimizer. This could be the string identifier of an existing optimizer or an instance of the `Optimizer` class；A loss function. This is the objective that the model will try to minimize. It can be the string identifier of an existing loss function or it can be an objective function；A list of metrics. A metric could be the string identifier of an existing metric or a custom metric function.

#### Guide to the Functional `API`

The `Keras` functional `API` is the way to go for defining complex models, such as multi-output models, directed acyclic graphs, or models with shared layers. With the functional `API`, it is easy to reuse trained models: you can treat any model as if it were a layer, by calling it on a tensor. Note that by calling a model you aren't just reusing the *architecture* of the model, you are also reusing its weights.

Whenever you are calling a layer on some input, you are creating a new tensor (the output of the layer), and you are adding a "node" to the layer, linking the input tensor to the output tensor. When you are calling the same layer multiple times, that layer owns multiple nodes indexed as 0, 1, 2...

The same is true for the properties `input_shape` and `output_shape`: as long as the layer has only one node, or as long as all nodes have the same input/output shape, then the notion of "layer output/input shape" is well defined, and that one shape will be returned by `layer.output_shape`/`layer.input_shape`. But if, for instance, you apply the same `Conv2D`layer to an input of shape `(32, 32, 3)`, and then to an input of shape `(64, 64, 3)`, the layer will have multiple input/output shapes, and you will have to fetch them by specifying the index of the node they belong to.

### Models

These models have a number of methods and attributes in common:

- `model.layers` is a flattened list of the layers comprising the model.
- `model.inputs` is the list of input tensors of the model.
- `model.outputs` is the list of output tensors of the model.
- `model.summary()` prints a summary representation of your model. 
- `model.get_config()` returns a dictionary containing the configuration of the model. 
- `model.get_weights()` returns a list of all weight tensors in the model, as Numpy arrays.
- `model.set_weights(weights)` sets the values of the weights of the model, from a list of `Numpy` arrays.
- `model.to_json()` returns a representation of the model as a `JSON` string. Note that the representation does not include the weights, only the architecture.
- `model.to_yaml()` returns a representation of the model as a `YAML` string. Note that the representation does not include the weights, only the architecture.
- `model.save_weights(filepath)` saves the weights of the model as a `HDF5` file.
- `model.load_weights(filepath, by_name=False)` loads the weights of the model from a `HDF5` file. By default, the architecture is expected to be unchanged. To load weights into a different architecture, use `by_name=True` to load only those layers with the same name.

In addition to these two types of models, you may create your own fully-customizable models by `subclassing` the `Model` class and implementing your own forward pass in the `call` method. **Layers** are defined in `__init__(self, ...)`, and the forward pass is specified in `call(self, inputs)`. In `call`, you may specify custom losses by calling `self.add_loss(loss_tensor)`.

In `subclassed` models, the model's topology is defined as Python code (rather than as a static graph of layers). That means the model's topology cannot be inspected or serialized. As a result, the following methods and attributes are not available for `subclassed` models:

- `model.inputs` and `model.outputs`.
- `model.to_yaml()` and `model.to_json()`
- `model.get_config()` and `model.save()`.

#### Sequential

`compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)`

**loss_weights**: Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs. The loss value that will be minimized by the model will then be the *weighted sum* of all individual losses, weighted by the `loss_weights` coefficients.

`fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1)`

- **x**: `Numpy` array of training data (if the model has a single input), or list of `Numpy` arrays (if the model has multiple inputs). If input layers in the model are named, you can also pass a dictionary mapping input names to `Numpy` arrays. `x` can be `None` (default) if feeding from framework-native tensors 
- **batch_size**: Integer or `None`. Number of samples per gradient update.
- **epochs**: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. 
- **callbacks**: List of `keras.callbacks.Callback` instances. List of callbacks to apply during training and validation.
- **validation_split**: Float between 0 and 1. Fraction of the training data to be used as validation data. 
- **validation_data**: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)`on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
- **shuffle**: Boolean (whether to shuffle the training data before each epoch) .
- **class_weight**: Optional dictionary mapping class `indices` (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
- **sample_weight**: Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only). You can either pass a flat (1D) Numpy array with the same length as the input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with shape `(samples, sequence_length)`, to apply a different weight to every timestep of every sample. In this case you should make sure to specify `sample_weight_mode="temporal"` in `compile()`.
- **initial_epoch**: Integer. Epoch at which to start training (useful for resuming a previous training run).
- **steps_per_epoch**: Integer or `None`. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as `TensorFlow` data tensors, the default `None` is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.

`evaluate(x=**None**, y=**None**, batch_size=**None**, verbose=1, sample_weight=**None**, steps=**None**, callbacks=**None**)`

`predict(x, batch_size=**None**, verbose=0, steps=**None**, callbacks=**None**)`

`get_layer(name=None, index=None)`: Retrieves a layer based on either its name (unique) or index. If `name` and `index` are both provided, `index` will take precedence. `Indices` are based on order of horizontal graph traversal

### Layers

All `Keras` layers have a number of methods in common:

- `layer.get_weights()`: returns the weights of the layer as a list of `Numpy` arrays.
- `layer.set_weights(weights)`: sets the weights of the layer from a list of Numpy arrays (with the same shapes as the output of `get_weights`).
- `layer.get_config()`: returns a dictionary containing the configuration of the layer. 

If a layer has a single node, you can get its input tensor, output tensor, input shape and output shape via:

- `layer.input`
- `layer.output`
- `layer.input_shape`
- `layer.output_shape`

If the layer has multiple nodes, you can use the following methods:

- `layer.get_input_at(node_index)`
- `layer.get_output_at(node_index)`
- `layer.get_input_shape_at(node_index)`
- `layer.get_output_shape_at(node_index)`

#### Core layers

- `use_bias`: Boolean, whether the layer uses a bias vector.
- `kernel_initializer`: Initializer for the `kernel` weights matrix
- `bias_initializer`: Initializer for the bias vector 
- `activation`: Activation function to use
- `kernel_regularizer`: Regularizer function applied to the `kernel` weights matrix 
- `bias_regularizer`: Regularizer function applied to the bias vector 
- `activity_regularizer`: Regularizer function applied to the output of the layer
- `kernel_constraint`: Constraint function applied to the `kernel` weights matrix 
- `bias_constraint`: Constraint function applied to the bias vector 

| 函数                                         | 作用                                                         | 输入                      | 输出                                                         |
| -------------------------------------------- | ------------------------------------------------------------ | ------------------------- | ------------------------------------------------------------ |
| `Dense(units)`                               | `Dense` implements the operation: `output = activation(dot(input, kernel) + bias)` | `(batch_size, input_dim)` | `(batch_size, units)`.                                       |
| `Activation(activation)`                     | Applies an activation function to an output.                 | Arbitrary                 | Same shape as input                                          |
| `Dropout(rate, noise_shape=None, seed=None)` | Dropout consists in randomly setting a fraction `rate` of input units to 0 at each update during training time, which helps prevent `overfitting`. |                           |                                                              |
| `Flatten(data_format=None)`                  | Flattens the input. Does not affect the batch size.          |                           |                                                              |
| `Reshape(target_shape)`                      | Reshapes an output to a certain shape.                       | Arbitrary                 | `(batch_size,) + target_shape`                               |
| `Permute(dims)`                              | Permutes the dimensions of the input according to a given pattern. | Arbitrary                 | Same as the input shape, but with the dimensions re-ordered according to the specified pattern |
| `RepeatVector(n)`                            | Repeats the input n times.                                   | `(num_samples, features)` | `(num_samples, n, features)`                                 |
| `ActivityRegularization(l1=0.0, l2=0.0)`     | Layer that applies an update to the cost function based input activity. | Arbitrary                 | Same as the input shape                                      |
| `Masking(mask_value=0.0)`                    |                                                              |                           |                                                              |
| `SpatialDropout1/2/3D(rate)`                 | This version performs the same function as Dropout, however it drops entire `1D` feature maps instead of individual elements. If adjacent frames within feature maps are strongly correlated then regular dropout will not regularize the activations and will otherwise just result in an effective learning rate decrease. In this case, `SpatialDropout1D` will help promote independence between feature maps and should be used instead. |                           |                                                              |

#### Convolutional layers

| 函数                                                         | 作用                                                         | 输入                                                 | 输出                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------------- | ------------------------------------------------------------ |
| `Conv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1)` | This layer creates a convolution kernel that is convolved with the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs. | `(batch, steps, channels)`                           | `(batch, new_steps, filters)`                                |
| `Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))` | This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. | `(batch, channels, rows, cols)`                      | `(batch, filters, new_rows, new_cols)`                       |
| `SeparableConv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, depth_multiplier=1)` | Separable convolutions consist in first performing a `depthwise` spatial convolution (which acts on each input channel separately) followed by a `pointwise` convolution which mixes together the resulting output channels. | `(batch, channels, steps)`                           | `(batch, filters, new_steps)`                                |
| `SeparableConv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), depth_multiplier=1)` |                                                              | `(batch, channels, rows, cols)`                      | `(batch, filters, new_rows, new_cols)`                       |
| `DepthwiseConv2D(kernel_size, strides=(1, 1), padding='valid', depth_multiplier=1, data_format=**None**)` | `Depthwise` Separable convolutions consists in performing just the first step in a `depthwise` spatial convolution (which acts on each input channel separately). | `(batch, channels, rows, cols)`                      | `(batch, filters, new_rows, new_cols)`                       |
| `Conv2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', output_padding=None, data_format=None, dilation_rate=(1, 1))` | from something that has the shape of the output of some convolution to something that has the shape of its input while maintaining a connectivity pattern that is compatible with said convolution. | `(batch, channels, rows, cols)`                      | `(batch, filters, new_rows, new_cols)`                       |
| `Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=**None**, dilation_rate=(1, 1, 1))` |                                                              | `(batch, channels, conv_dim1, conv_dim2, conv_dim3)` | `(batch, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)` |
| `Conv3DTranspose(filters, kernel_size, strides=(1, 1, 1), padding='valid', output_padding=None, data_format=None)` |                                                              | `(batch, channels, depth, rows, cols)`               | `(batch, filters, new_depth, new_rows, new_cols)`            |

##### Pooling

| 函数                                                         | 输入                                                         | 输出                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `MaxPooling1D(pool_size=2, strides=**None**, padding='valid', data_format='channels_last')` | `(batch_size, steps, features)`                              | `(batch_size, downsampled_steps, features)`                  |
| `MaxPooling2D(pool_size=(2, 2))`                             | `(batch_size, rows, cols, channels)`                         | `(batch_size, pooled_rows, pooled_cols, channels)`           |
| `MaxPooling3D(pool_size=(2, 2, 2))`                          | `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)` | `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)` |
| `AveragePooling1/2/3D(pool_size=2)`                          |                                                              |                                                              |
| `GlobalMaxPooling1D(data_format='channels_last')`            | `(batch_size, steps, features)`                              | `(batch_size, features)`                                     |
| `GlobalAveragePooling2D(data_format=**None**)`               | `(batch_size, rows, cols, channels)`                         | `(batch_size, channels)`                                     |
| `GlobalAveragePooling2D(data_format=**None**)`               |                                                              |                                                              |

##### Locally-Connected layers

The `LocallyConnected1D` layer works similarly to the `Conv1D` layer, except that weights are unshared, that is, a different set of filters is applied at each different patch of the input.

`LocallyConnected1/2/3D(filters, kernel_size, strides=(1, 1), padding='valid')`

#### Recurrent layers

`RNN(cell, return_sequences=**False**, return_state=**False**, go_backwards=**False**, stateful=**False**, unroll=**False**)`

- **cell**: A `RNN` cell instance. A `RNN` cell is a class that has: a `call(input_at_t, states_at_t)` method, returning `(output_at_t, states_at_t_plus_1)`. a `state_size` attribute. This can be a single integer (single state) in which case it is the size of the recurrent state. a `output_size` attribute. This can be a single integer or a `TensorShape`, which represent the shape of the output. 
- **return_sequences**: Boolean. Whether to return the last output in the output sequence, or the full sequence.
- **return_state**: Boolean. Whether to return the last state in addition to the output.
- **go_backwards**: Boolean (default False). If True, process the input sequence backwards and return the reversed sequence.
- `stateful`: Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.
- **input_dim**: dimensionality of the input (integer). 
- **input_length**: Length of input sequences, to be specified when it is constant. This argument is required if you are going to connect `Flatten` then `Dense`layers upstream 

Input shape: `3D` tensor with shape `(batch_size, timesteps, input_dim)`. Output shape: if `return_state`: a list of tensors. The first tensor is the output. The remaining tensors are the last states, each with shape `(batch_size, units)`. if `return_sequences`: `3D` tensor with shape `(batch_size, timesteps, units)`. else, `2D` tensor with shape `(batch_size, units)`.

This layer supports masking for input data with a variable number of `timesteps`. To introduce masks to your data, use an Embedding layer with the `mask_zero` parameter set to `True`.

| 函数                                                         | 输入                                    | 输出                                                         |
| ------------------------------------------------------------ | --------------------------------------- | ------------------------------------------------------------ |
| `SimpleRNN(units, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)` |                                         |                                                              |
| `GRU(units, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=False)` |                                         |                                                              |
| `LSTM(units, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)` |                                         |                                                              |
| `ConvLSTM2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), return_sequences=False, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0)` | `(samples, time, channels, rows, cols)` | `(samples, time, filters, output_row, output_col)`or `(samples, filters, output_row, output_col)` |

`ConvLSTM2DCell(filters, kernel_size, strides=(1, 1), dropout=0.0, recurrent_dropout=0.0), SimpleRNNCell(units, dropout=0.0, recurrent_dropout=0.0), GRUCell(units, dropout=0.0, recurrent_dropout=0.0, implementation=1, reset_after=**False**), LSTMCell(units, dropout=0.0, recurrent_dropout=0.0, implementation=1)`

#### Embedding layers

`keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)`

- **input_dim**: int > 0. Size of the vocabulary.
- **output_dim**: int >= 0. Dimension of the dense embedding.
- **mask_zero**: Whether or not the input value 0 is a special "padding" value that should be masked out. This is useful when using recurrent layers which may take variable length input. If this is `True` then all subsequent layers in the model need to support masking or an exception will be raised. If mask_zero is set to True, as a consequence, index 0 cannot be used in the vocabulary (input_dim should equal size of vocabulary + 1).
- **input_length**: Length of input sequences, when it is constant. This argument is required if you are going to connect `Flatten` then `Dense` layers upstream.

Input shape: `2D` tensor with shape: `(batch_size, sequence_length)`.

Output shape: `3D` tensor with shape: `(batch_size, sequence_length, output_dim)`.

##### Merge layers

| 函数                             | 作用                                                         |
| -------------------------------- | ------------------------------------------------------------ |
| `Add()`                          | Layer that adds a list of inputs. It takes as input a list of tensors, all of the same shape, and returns a single tensor (also of the same shape) |
| `Subtract()`                     | It takes as input a list of tensors of size 2, both of the same shape, and returns a single tensor, (inputs[0] - inputs[1]), also of the same shape. |
| `Average()`                      | `It takes as input a list of tensors, all of the same shape, and returns a single tensor (also of the same shape).` |
| `Maximum()/Minimum()`            | It takes as input a list of tensors, all of the same shape, and returns a single tensor (also of the same shape). |
| `Concatenate(axis=-1)`           | It takes as input a list of tensors, all of the same shape except for the concatenation axis, and returns a single tensor, the concatenation of all inputs. |
| `Dot(axes, normalize=**False**)` |                                                              |

#### Activations layers

| 激活层                                                       | 表达式                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `LeakyReLU(alpha=0.3)`                                       | `f(x) = alpha * x for x < 0`, `f(x) = x for x >= 0`          |
| `PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)` | `f(x) = alpha * x for x < 0`, `f(x) = x for x >= 0`, where `alpha` is a learned array with the same shape as x. |
| `ELU(alpha=1.0)`                                             | `f(x) =  alpha * (exp(x) - 1.) for x < 0`, `f(x) = x for x >= 0` |
| `ThresholdedReLU(theta=1.0)`                                 | `f(x) = x for x > theta`, `f(x) = 0 otherwise`               |
| `Softmax(axis=-1)`                                           | `Softmax` activation function.                               |
| `ReLU(max_value=**None**, negative_slope=0.0, threshold=0.0)` | `f(x) = max_value` for `x >= max_value`, `f(x) = x` for `threshold <= x < max_value`, `f(x) = negative_slope * (x - threshold)` |

Activations can either be used through an `Activation` layer, or through the `activation`argument supported by all forward layers

| 激活函数                                | 表达式                                                       |
| --------------------------------------- | ------------------------------------------------------------ |
| `keras.activations.softmax(x, axis=-1)` | `Softmax` activation function.                               |
| `elu(x, alpha=1.0)`                     | `x` if `x > 0` and `alpha * (exp(x)-1)` if `x < 0`.          |
| `selu(x)`                               | `scale*elu(x, alpha)`                                        |
| `softplus(x)`                           | `log(exp(x) + 1)`                                            |
| `softsign(x)`                           | `x / (abs(x) + 1)`                                           |
| `hard_sigmoid(x)`                       | `0` if `x < -2.5` `1` if `x > 2.5` `0.2 * x + 0.5` if `-2.5 <= x <= 2.5`. |
| `sigmoid(x), tanh(x)`                   |                                                              |

#### Normalization layers

`BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, moving_mean_initializer='zeros', )`

`axis`: Integer, the axis that should be normalized. `momentum`: Momentum for the moving mean and the moving variance. `epsilon`: Small float added to variance to avoid dividing by zero. `center`: If True, add offset of `beta` to normalized tensor. If False, `beta` is ignored. `scale`: If True, multiply by `gamma`. If False, `gamma` is not used.

#### Noise layers

#### Layer wrappers

| 函数                                                         | 作用                                                         | 变量                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `TimeDistributed(layer)`                                     | This wrapper applies a layer to every temporal slice of an input. The input should be at least `3D`, and the dimension of index one will be considered to be the temporal dimension：时间维度. | `layer`: a layer instance.                                   |
| `Bidirectional(layer, merge_mode='concat', weights=**None**)` | Bidirectional wrapper for `RNNs`.                            | `layer`: `Recurrent` instance. `merge_mode`: Mode by which outputs of the forward and backward `RNNs` will be combined.  `weights`: Initial weights to load in the Bidirectional model |

#### writing your own `Keras` layers

There are only three methods you need to implement:

- `build(input_shape)`: this is where you will define your weights. This method must set `self.built = True` at the end, which can be done by calling `super([Layer], self).build()`.
- `call(x)`: this is where the layer's logic lives. Unless you want your layer to support masking, you only have to care about the first argument passed to `call`: the input tensor.
- `compute_output_shape(input_shape)`: in case your layer modifies the shape of its input, you should specify here the shape transformation logic. This allows `Keras` to do automatic shape inference.

It is also possible to define `Keras` layers which have multiple input tensors and multiple output tensors. To do this, you should assume that the inputs and outputs of the methods `build(input_shape)`, `call(x)` and `compute_output_shape(input_shape)` are lists.

### Preprocessing

#### Text Preprocessing

#### Image Preprocessing

#### Losses

| 损失函数名                                        | 表达式         |
| ------------------------------------------------- | -------------- |
| `keras.losses.mean_squared_error(y_true, y_pred)` |                |
| `mean_absolute_error(y_true, y_pred)`             |                |
| `mean_absolute_percentage_error(y_true, y_pred)`  |                |
| `mean_squared_logarithmic_error(y_true, y_pred)`  |                |
| `squared_hinge(y_true, y_pred)`                   |                |
| `hinge(y_true, y_pred)`                           |                |
| `categorical_hinge(y_true, y_pred)`               |                |
| `logcosh(y_true, y_pred)`                         | `log(cosh(x))` |
| `categorical_crossentropy(y_true, y_pred)`        |                |
| `sparse_categorical_crossentropy(y_true, y_pred)` |                |
| `binary_crossentropy(y_true, y_pred)`             |                |
| `kullback_leibler_divergence(y_true, y_pred)`     |                |
| `cosine_proximity(y_true, y_pred)`                |                |

A metric is a function that is used to judge the performance of your model. Metric functions are to be supplied in the `metrics` parameter when a model is compiled.

`keras.metrics.binary_accuracy(y_true, y_pred)`, `categorical_accuracy(y_true, y_pred)`, `sparse_categorical_accuracy(y_true, y_pred)`, `top_k_categorical_accuracy(y_true, y_pred, k=5)`, `sparse_top_k_categorical_accuracy(y_true, y_pred, k=5)`

#### Optimizers

The parameters `clipnorm` and `clipvalue` can be used with all optimizers to control gradient clipping

1. Stochastic gradient descent optimizer: `keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)`
2. `RMSProp` optimizer: `RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)`
3. `Adagrad` optimizer: `Adagrad(lr=0.01, epsilon=**None**, decay=0.0)`
4. `Adadelta` optimizer: `Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)`
5. `Adam` optimizer: `Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=**None**, decay=0.0, amsgrad=False)`
6. `Adamax`: `Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=**None**, decay=0.0)`
7. `Nesterov Adam` optimizer: `Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)`

#### Callbacks

A callback is a set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states and statistics of the model during training. You can pass a list of callbacks (as the keyword argument `callbacks`) to the `.fit()` method of the `Sequential` or `Model` classes. The relevant methods of the callbacks will then be called at each stage of the training.

#### Applications

`Keras` Applications are deep learning models that are made available alongside `pre-trained` weights. These models can be used for prediction, feature extraction, and fine-tuning. Weights are downloaded automatically when instantiating a model. They are stored at `~/.keras/models/`. All of these architectures are compatible with all the `backends`, and upon instantiation the models will be built according to the image data format set in your `Keras` configuration file at `~/.keras/keras.json`. For instance, if you have set `image_data_format=channels_last`, then any model loaded from this repository will get built according to the `TensorFlow` data format convention, "Height-Width-Depth".

`keras.applications.xception.Xception(include_top=**True**, weights='imagenet', input_tensor=**None**, input_shape=**None**, pooling=**None**, classes=1000)`

- include_top: whether to include the fully-connected layer at the top of the network.
- weights: one of `None` (random initialization) or `'imagenet'` (pre-training on ImageNet).
- input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.
- input_shape: optional shape tuple, only to be specified if `include_top` is `False`(otherwise the input shape has to be `(299, 299, 3)`. It should have exactly 3 inputs channels, and width and height should be no smaller than 71. E.g. `(150, 150, 3)` would be one valid value.
- pooling: Optional pooling mode for feature extraction when`include_top` is `False`
  - `None` means that the output of the model will be the `4D` tensor output of the last `convolutional` layer.
  - `'avg'` means that global average pooling will be applied to the output of the last convolutional layer, and thus the output of the model will be a 2D tensor.
  - `'max'` means that global max pooling will be applied.
- classes: optional number of classes to classify images into, only to be specified if `include_top` is `True`, and if no `weights` argument is specified.

Returns; A `Keras` `Model` instance.

#### Initializers

Initializations define the way to set the initial random weights of `Keras` layers. The keyword arguments used for passing initializers to layers will depend on the layer. Usually it is simply `kernel_initializer` and `bias_initializer`

| 函数                                                         | 作用                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `keras.initializers.Zeros()`                                 | Initializer that generates tensors initialized to 0.         |
| `Ones()`                                                     | Initializer that generates tensors initialized to 1.         |
| `Constant(value = 0)`                                        | Initializer that generates tensors initialized to a constant value. |
| `RandomNormal(mean=0.0, stddev=0.05, seed=None)`             | Initializer that generates tensors with a normal distribution. |
| `RandomUniform(minval=-0.05, maxval=0.05, seed=None)`        | Initializer that generates tensors with a uniform distribution. |
| `TruncatedNormal(mean=0.0, stddev=0.05, seed=None)`          | Initializer that generates a truncated normal distribution.  |
| `VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)` | Initializer capable of adapting its scale to the shape of weights. |
| `he_uniform(seed=**None**)`                                  | draws samples from a uniform distribution within [-limit, limit] where `limit` is `sqrt(6 / fan_in)` |
| `lecun_uniform(seed=None)`                                   | draws samples from a uniform distribution within [-limit, limit] where `limit` is `sqrt(3 / fan_in)` |
| `glorot_normal(seed=**None**)`                               | draws samples from a truncated normal distribution centered on 0 with `stddev = sqrt(2 / (fan_in + fan_out))` |
| `glorot_uniform(seed=**None**)`                              | draws samples from a uniform distribution within [-limit, limit] where `limit` is `sqrt(6 / (fan_in + fan_out))` |
| `he_normal(seed=**None**)`                                   | draws samples from a truncated normal distribution centered on 0 with `stddev = sqrt(2 / fan_in)` |
| `lecun_normal(seed=**None**)`                                | draws samples from a truncated normal distribution centered on 0 with `stddev = sqrt(1 / fan_in)` |

#### Regularizers

`Regularizers` allow to apply penalties on layer parameters or layer activity during optimization. These penalties are incorporated in the loss function that the network optimizes.

These layers expose 3 keyword arguments:

- `kernel_regularizer`: instance of `keras.regularizers.Regularizer`
- `bias_regularizer`: instance of `keras.regularizers.Regularizer`
- `activity_regularizer`: instance of `keras.regularizers.Regularizer`

`keras.regularizers.l1(0.)
keras.regularizers.l2(0.)
keras.regularizers.l1_l2(l1=0.01, l2=0.01)`

#### Constraints

Functions from the `constraints` module allow setting constraints on network parameters during optimization.

These layers expose 2 keyword arguments:

- `kernel_constraint` for the main weights matrix
- `bias_constraint` for the bias.

| 函数                                                         | 作用                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `keras.constraints.MaxNorm(max_value=2, axis=0)`             | Constrains the weights incident to each hidden unit to have a norm less than or equal to a desired value. |
| `NonNeg()`                                                   | Constrains the weights to be non-negative.                   |
| `MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0, axis=0)` | Constrains the weights incident to each hidden unit to have the norm between a lower bound and an upper bound. |
| `UnitNorm(axis=0)`                                           | Constrains the weights incident to each hidden unit to have unit norm. |

#### Utils

