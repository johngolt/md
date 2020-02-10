#### High Level `APIs`

##### `Keras`

##### Eager Execution

##### Importing Data

#### Estimators

##### `Premade Estimators`

##### Checkpoints

##### Feature Columns

##### Datasets for Estimators

##### Creating Custom Estimators

#### Low level `APIs`

This guide gets you started programming in the low-level `TensorFlow APIs` , showing you how to: Manage your own `TensorFlow` program and `TensorFlow` runtime, instead of relying on Estimators to manage them. Run `TensorFlow` operations. Use high level components in this low level environment. Build your own training loop.

We recommend using the higher level `APIs` to build models when possible. Knowing `TensorFlow` Core is valuable for the following reasons: Experimentation and debugging are both more straight forward when you can use low level `TensorFlow` operations directly. It gives you a mental model of how things work internally when using the higher level `APIs`.

Some `TensorFlow` functions return `tf.Operations` instead of `tf.Tensors`. The result of calling `run` on an Operation is `None`. You run an operation to cause a side-effect, not to retrieve a value.

##### Tensors

 A tensor consists of a set of primitive values shaped into an array of any number of dimensions. A tensor's **rank** is its number of dimensions, while its **shape** is a tuple of integers specifying the array's length along each dimension. `TensorFlow` uses `numpy` arrays to represent tensor **values**.

A `tf.Tensor` object represents a partially defined computation that will eventually produce a value. `TensorFlow` programs work by first building a graph of `tf.Tensor`objects, detailing how each tensor is computed based on the other available tensors and then by running parts of this graph to achieve the desired results.

With the exception of `tf.Variable`, the value of a tensor is immutable, which means that in the context of a single execution tensors only have a single value. However, evaluating the same tensor twice can return different values

The **shape** of a tensor is the number of elements in each dimension. `TensorFlow` automatically infers shapes during graph construction. These inferred shapes might have known or unknown rank. If the rank is known, the sizes of each dimension might be known or unknown.

This can be done by reading the `shape` property of a `tf.Tensor`object. This method returns a `TensorShape` object, which is a convenient way of representing partially-specified shapes (since, when building the graph, not all shapes will be fully known). It is also possible to get a `tf.Tensor`that will represent the fully-defined shape of another `tf.Tensor`at runtime. This is done by calling the `tf.shape`operation. This way, you can build a graph that manipulates the shapes of tensors by building other tensors that depend on the dynamic shape of the input `tf.Tensor`.

When creating a `tf.Tensor`from a python object you may optionally specify the `datatype`. If you don't, `TensorFlow` chooses a `datatype` that can represent your data. `TensorFlow` converts Python integers to `tf.int32` and python floating point numbers to `tf.float32`. Otherwise `TensorFlow` uses the same rules `numpy` uses when converting to arrays.

##### Variables

A `TensorFlow` variable is the best way to represent shared, persistent state manipulated by your program.Variables are manipulated via the `tf.Variable` class. A` tf.Variable` represents a tensor whose value can be changed by running ops on it. Unlike` tf.Tensor` objects, a `tf.Variable` exists outside the context of a single `session.run` call.

Internally, a `tf.Variable` stores a persistent tensor. Specific ops allow you to read and modify the values of this tensor. These modifications are visible across multiple` tf.Sessions`, so multiple workers can see the same values for a `tf.Variable`.

The best way to create a variable is to call the `tf.get_variable` function. This function requires you to specify the Variable's name. This name will be used by other replicas to access the same variable, as well as to name this variable's value when `checkpointing` and exporting models. `tf.get_variable` also allows you to reuse a previously created variable of the same name, making it easy to define models which reuse layers.

Because disconnected parts of a `TensorFlow` program might want to create variables, it is sometimes useful to have a single way to access all of them. For this reason `TensorFlow` provides collections, which are named lists of tensors or other objects. By default every `tf.Variable` gets placed in the following two collections: `tf.GraphKeys.GLOBAL_VARIABLES` --- variables that can be shared across multiple devices,
`tf.GraphKeys.TRAINABLE_VARIABLES` --- variables for which `TensorFlow` will calculate gradients. If you don't want a variable to be trainable, add it to the `tf.GraphKeys.LOCAL_VARIABLES` collection instead.  You can also use your own collections. Any string is a valid collection name, and there is no need to explicitly create a collection. To add a variable to a collection after creating the variable, call `tf.add_to_collection`. 

Before you can use a variable, it must be initialized. If you are programming in the low-level `TensorFlow API`, you must explicitly initialize the variables. To initialize all trainable variables in one go, before training starts, call `tf.global_variables_initializer()`. This function returns a single operation responsible for initializing all variables in the `tf.GraphKeys.GLOBAL_VARIABLES` collection. Running this operation initializes all variables. Note that by default `tf.global_variables_initializer` does not specify the order in which variables are initialized. Therefore, if the initial value of a variable depends on another variable's value, it's likely that you'll get an error. Any time you use the value of a variable in a context in which not all variables are initialized, it is best to use `variable.initialized_value()` instead of variable

`TensorFlow` supports two ways of sharing variables: Explicitly passing `tf.Variable` objects around. Implicitly wrapping `tf.Variable` objects within `tf.variable_scope` objects.
While code which explicitly passes variables around is very clear, it is sometimes convenient to write `TensorFlow` functions that implicitly use variables in their implementations.Variable scopes allow you to control variable reuse when calling functions which implicitly create and use variables. They also allow you to name your variables in a hierarchical and understandable way.
If you do want the variables to be shared, you have two options. First, you can create a scope with the same name using `reuse=True` You can also call `scope.reuse_variables()` to trigger a reuse
Since depending on exact string names of scopes can feel dangerous, it's also possible to initialize a variable scope based on another one

##### Graphs and Sessions

`TensorFlow` uses a `dataflow` graph to represent your computation in terms of the dependencies between individual operations. This leads to a low-level programming model in which you first define the `dataflow` graph, then create a `TensorFlow` session to run parts of the graph across a set of local and remote devices.

A `tf.Graph` contains two relevant kinds of information: **Graph structure**. The nodes and edges of the graph, indicating how individual operations are composed together, but not prescribing how they should be used. The graph structure is like assembly code: inspecting it can convey some useful information, but it does not contain all of the useful context that source code conveys. **Graph collections**. `TensorFlow` provides a general mechanism for storing collections of `metadata` in a `tf.Graph`. The `tf.add_to_collection` function enables you to associate a list of objects with a key (where `tf.GraphKeys` defines some of the standard keys), and `tf.get_collection` enables you to look up all objects associated with a key. Many parts of the TensorFlow library use this facility.
Most `TensorFlow` programs start with a `dataflow` graph construction phase. In this phase, you invoke `TensorFlow API` functions that construct new `tf.Operation` (node) and `tf.Tensor` (edge) objects and add them to a `tf.Graph` instance. `TensorFlow` provides a default graph that is an implicit argument to all `API` functions in the same context. 
A `tf.Graph` object defines a `namespace` for the `tf.Operation` objects it contains. `TensorFlow` automatically chooses a unique name for each operation in your graph, but giving operations descriptive names can make your program easier to read and debug. The `TensorFlow API` provides two ways to override the name of an operation: Each `API` function that creates a new `tf.Operation` or returns a new `tf.Tensor` accepts an optional name argument.  The `tf.name_scope` function makes it possible to add a name scope prefix to all operations created in a particular context. The current name scope prefix is a "/"-delimited list of the names of all active `tf.name_scope` context managers. If a name scope has already been used in the current context, `TensorFlow` appends "_1", "_2", and so on.
Note that `tf.Tensor` objects are implicitly named after the `tf.Operation` that produces the tensor as output. A tensor name has the form "<OP_NAME>:<i>" where: "<OP_NAME>" is the name of the operation that produces it. "<i>" is an integer representing the index of that tensor among the operation's outputs.

If you want your `TensorFlow` program to use multiple different devices, the `tf.device` function provides a convenient way to request that all operations created in a particular context are placed on the same device

a tensor-like object in place of a `tf.Tensor`, and implicitly convert it to a `tf.Tensor` using the `tf.convert_to_tensor` method. Tensor-like objects include elements of the following types: `tf.Tensor`; 
`tf.Variable`;`numpy.ndarray`; list (and lists of tensor-like objects); Scalar Python types: `bool, float, int, str`.

A `tf.Session` object provides access to devices in the local machine, and remote devices using the distributed `TensorFlow` runtime. It also caches information about your `tf.Graph` so that you can efficiently run the same computation multiple times.

Since a `tf.Session` owns physical resources, it is typically used as a context manager that automatically closes the session when you exit the block. It is also possible to create a session without using a with block, but you should explicitly call `tf.Session.close` when you are finished with it to free the resources.

`tf.Session.init` accepts three optional arguments: target. If this argument is left empty, the session will only use devices in the local machine; `graph`. By default, a new `tf.Sessio`n will be bound to---and only able to run operations in---the current default graph. If you are using multiple graphs in your program , you can specify an explicit `tf.Graph` when you construct the session; `config`. This argument allows you to specify a `tf.ConfigProto` that controls the behavior of the session. 

The `tf.Session.run` method is the main mechanism for running a `tf.Operation` or evaluating a `tf.Tensor`. You can pass one or more `tf.Operation` or `tf.Tensor` objects to `tf.Session.run`, and `TensorFlow` will execute the operations that are needed to compute the result. `tf.Session.run` requires you to specify a list of fetches, which determine the return values, and may be a `tf.Operation`, a `tf.Tensor`, or a tensor-like type. These fetches determine what subgraph of the overall `tf.Graph` must be executed to produce the result: this is the subgraph that contains all operations named in the fetch list, plus all operations whose outputs are used to compute the value of the fetches.  `tf.Session.run` also optionally takes a dictionary of feeds, which is a mapping from `tf.Tensor` objects (typically `tf.placeholder` tensors) to values (typically Python scalars, lists, or `NumPy` arrays) that will be substituted for those tensors in the execution.

`TensorFlow` includes tools that can help you to understand the code in a graph. The graph visualizer is a component of `TensorBoard` that renders the structure of your graph visually in a browser. The easiest way to create a visualization is to pass a `tf.Graph` when creating the `tf.summary.FileWriter`

The default graph stores information about every `tf.Operation` and `tf.Tensor` that was ever added to it. If your program creates a large number of unconnected subgraphs, it may be more efficient to use a different `tf.Graph` to build each subgraph, so that unrelated state can be garbage collected. You can install a different `tf.Graph` as the default graph, using the `tf.Graph.as_default` context manager

##### Save and Restore

##### Control flow

##### Ragged Tensors

 Ragged tensors are the `TensorFlow` equivalent of nested variable-length lists. They make it easy to store and process data with non-uniform shapes, including: Variable-length features, such as the set of actors in a movie. Batches of variable-length sequential inputs, such as sentences or video clips. Hierarchical inputs, such as text documents that are subdivided into sections, paragraphs, sentences, and words. Individual fields in structured inputs, such as protocol buffers.

```python
queries = tf.ragged.constant([['Who', 'is', 'Dan', 'Smith'],['Pause'],
                              ['Will', 'it', 'rain', 'later', 'today']])
num_buckets = 1024
embedding_size = 4
embedding_table = tf.Variable(
    tf.truncated_normal([num_buckets, embedding_size],
                       stddev=1.0 / math.sqrt(embedding_size)))
word_buckets = tf.strings.to_hash_bucket_fast(queries, num_buckets)
word_embeddings = tf.ragged.map_flat_values(
    tf.nn.embedding_lookup, embedding_table, word_buckets)                  # ①

marker = tf.fill([queries.nrows(), 1], '#')
padded = tf.concat([marker, queries, marker], axis=1)                       # ②

bigrams = tf.string_join([padded[:, :-1], padded[:, 1:]], separator='+')    # ③
bigram_buckets = tf.strings.to_hash_bucket_fast(bigrams, num_buckets)
bigram_embeddings = tf.ragged.map_flat_values(
    tf.nn.embedding_lookup, embedding_table, bigram_buckets)                # ④

all_embeddings = tf.concat([word_embeddings, bigram_embeddings], axis=1)    # ⑤
avg_embedding = tf.reduce_mean(all_embeddings, axis=1)                      # ⑥
```



![](./picture/rag.png)

If you need to perform an `elementwise` transformation to the values of a `RaggedTensor`, you can use `tf.ragged.map_flat_values`, which takes a function plus one or more arguments, and applies the function to transform the `RaggedTensor's` values. The simplest way to construct a ragged tensor is using `tf.ragged.constant`, which builds the `RaggedTensor` corresponding to a given nested Python list.
Ragged tensors can also be constructed by pairing flat values tensors with row-partitioning tensors indicating how those values should be divided into rows, using factory `classmethods` such as `tf.RaggedTensor.from_value_rowids`, `tf.RaggedTensor.from_row_lengths`, and `tf.RaggedTensor.from_row_splits`. If you know which row each value belongs in, then you can build a `RaggedTensor` using a `value_rowids` row-partitioning tensor:

![](./picture/1/22.png)

If you know how long each row is, then you can use a row_lengths row-partitioning tensor:

![](./picture/1/23.png)

If you know the index where each row starts and ends, then you can use a row_splits row-partitioning tensor:

![](./picture/1/24.png)

As with normal Tensors, the values in a `RaggedTensor` must all have the same type; and the values must all be at the same nesting depth (the rank of the tensor). The outermost dimension of a ragged tensor is always uniform, since it consists of a single slice. In addition to the uniform outermost dimension, ragged tensors may also have uniform inner dimensions. For example, we might store the word embeddings for each word in a batch of sentences using a ragged tensor with shape `[num_sentences, (num_words), embedding_size]`, where the parentheses around `(num_words)` indicate that the dimension is ragged.
Dimensions whose slices all have the same length are called uniform dimensions.
The shape of a ragged tensor is currently restricted to have the following form: A single uniform dimension; Followed by one or more ragged dimensions; Followed by zero or more uniform dimensions.

![](./picture/1/25.png)

Ragged tensors are encoded using the `RaggedTensor` class. Internally, each `RaggedTensor` consists of: A values tensor, which concatenates the variable-length rows into a flattened list. A row_splits vector, which indicates how those flattened values are divided into rows. In particular, the values for row rt[i] are stored in the slice rt.values[rt.row_splits[i]:rt.row_splits[i+1]].

![](./picture/1/26.png)

A ragged tensor with multiple ragged dimensions is encoded by using a nested `RaggedTensor` for the values tensor. Each nested `RaggedTensor` adds a single ragged dimension.

![](./picture/1/27.png)

Ragged tensors with uniform inner dimensions are encoded by using a multidimensional `tf.Tensor` for values.

![](./picture/1/28.png)

#### Other

##### 构建图

在构建阶段，`op`d的执行步骤被描述成一个图，在执行阶段，使用会话执行途中的`op`。构建图的第一步是创建源`op`。源`op`不需要任何输入，源`op`的输出被传递给其他`op`做运算。`op`构造器的返回值代表被构造出的`op`的输出，这些返回值可以传递给其他`op`做运算。`tf.placeholder`操作，定义传入图表中的`shape`参数，`shape`参数包括`batch_size`值，后续还会将实际的训练数据传入图中。

在为数据创建占位符之后，经过三个阶段的模式函数操作：`inference(), loss() and training()`。图就构建完成。`Inference()`：尽可能地构建好图，满足促使神经网络向前反馈并作出预测的要求，做到返回包括预测结果的`Tensor`；`Loss()`：往`inference`图中添加损失所需要的操作， 返回包含了损失值的`Tensor`。`training()`：往损失图中添加计算并应用梯度所需要的操作。

##### Tensor Transformations

| 函数                                                  | 作用 |
| ----------------------------------------------------- | ---- |
| `tf.string_to_number(string, tensor, out_type, name)` |      |
| `tf.to_float/int32/..(x, name)`                       |      |
| `tf.cast(x, dtype, name)`                             |      |

##### Shapes and Shaping

| 函数                                                         | 作用 |
| ------------------------------------------------------------ | ---- |
| `tf.shape/size/rank(input, name)`                            |      |
| `tf.reshape(tensor, shape, name)`                            |      |
| `tf.squeeze(input, squeeze_dims, name)`                      |      |
| `tf.expand_dims(input, dim, name)`                           |      |
| `tf.slice(input, begin, size, name)`                         |      |
| `tf.split(split_dim, num_split, value, name)`                |      |
| `tf.tile(input, multiples, name)`                            |      |
| `tf.pad(input, paddings, name)`                              |      |
| `tf.concat(concat_dim, values, name)`                        |      |
| `tf.pack(value, name)`                                       |      |
| `tf.unpack(value, num, name)`                                |      |
| `tf.transpose(a, perm = None, name)`                         |      |
| `tf.gather(params, indices, name)`                           |      |
| `tf.reverse(tensor, dims, name)`                             |      |
| `tf.reverse_sequence(input, seq_lengths, seq_dim, name)`     |      |
| `tf.dynamic_partition(data, partitions, num_partitions, name)` |      |
| `tf.dynamic_stitch(indices, data, name)`                     |      |

##### Math

| 函数                                                         | 作用 |
| ------------------------------------------------------------ | ---- |
| `tf.add_n(inputs, name)`                                     |      |
| `tf.reduce_sum(input_tensor, reduction_indices, keep_dims, name)` |      |
| `tf.segment_min(data, segment_ids, name)`                    |      |
| `tf.accumulate_n(inputs, shape, tensor_dtype, name)`         |      |
| `tf.argmin(input, dimension, name)`                          |      |
| `tf.where(input, name)`                                      | `    |
| `tf.invert_permutation(x, name)`                             |      |

##### Neural Network

| 函数                                                         | 作用 |
| ------------------------------------------------------------ | ---- |
| `tf.nn.dropout(x, keep_prob noise_shape, seed, name)`        |      |
| `tf.nn.bias_add(value, bias, name)`                          |      |
| `tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu, name)` |      |
| `tf.nn.depthwise_cov2d(input, filter, striders, padding, name)` |      |
| `tf.nn.separable_conv2d(input, depthwise_filter, pointwise_filter, strides, padding, name)` |      |
| `tf.nn.max_pool_with_argmax(input, ksize, strides, padding, Targmax, name)` |      |
| `tf.nn.l2_normalize(x, dim, epsilon=1e-12, name)`            |      |
| `tf.nn.moments(x, axes, name)`                               |      |
| `tf.nn.sigmoid_cross_entropy_with_logits(logits, targets, name)` |      |
| `tf.nn.embedding_lookup(params, ids, name)`                  |      |
| `tf.nn.softmax_cross_entropy_with_logits(logits, labels, name)` |      |
| `tf.nn.nce_loss(weights, biases, inputs, labels, num_sampled, num_classes, num_true=1, sampled_values,name)` |      |
| `tf.nn.sample_softmax_loss(weights, biases, inputs, labels, num_sampled, num_classes, num_true=1, sampled_values,name)` |      |

##### Variable

| 函数                                                         | 作用 |
| ------------------------------------------------------------ | ---- |
| `tf.initialize_all_variables()`                              |      |
| `tf.initialize_variables(var_list, name)`                    |      |
| `tf.assert_variables_initialized(var_list)`                  |      |
| `tf.train.latest_checkpoint(checkpoint_dir, latest_filename)` |      |
| `tf.train.get_checkpoint_state(checkpoint_dir, latest_filename)` |      |
| `tf.train.update_checkpoint_state(save_dir, model_checkpoint_path, all_model_checkpoint_paths, latest_filename)` |      |
| `tf.get_variable(name, shape, dtype=tf.float32, initializer, trainable=True, collections)` |      |
| `tf.get_variable_scope()`                                    |      |
| `tf.variable_scope(name_or_scope, reuse, initializer)`       |      |

###### Training

| 函数                                                         | 作用                            |
| ------------------------------------------------------------ | ------------------------------- |
| `tf.gradients(ys, xs, grad_ys=None, name, colocate_gradients_with_ops=False, gate_gradients=False, aggregation_method=None)` |                                 |
| `tf.clip_by_value(t, clip_value_min, clip_value_max, name)`  |                                 |
| `tf. clip_by_norm(t, clip_norm, name)`                       |                                 |
| `tf.clip_by_average_norm(t, clip_norm, name)`                |                                 |
| `tf.clip_by_global_norm(t_list, clip_norm, use_norm, name)`  |                                 |
| `tf.global_norm(t_list, name)`                               |                                 |
| `tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name)` |                                 |
| `tf.scalar_summary(tags, values, collections, name)`         |                                 |
| `tf.image_summary(tag, tensor, max_images, collections, name)` |                                 |
| `tf.histogram_summary(tag, values, collections, name)`       |                                 |
| `tf.merge_summary(inputs, collections, name)`                |                                 |
| `tf.merge_all_summary()`                                     |                                 |
| `tf.train.SummaryWriter`                                     | adding Summaries to Event Files |
| `tf.train.global_step(sess, global_step_tensor)`             |                                 |
| `tf.train.write_graph(graph_def, logdir, name, as_text=True)` |                                 |
| `tf.train.summary_iterator(path)`                            |                                 |

`Optimizer`: compute gradients for a loss and apply gradients to variables. Call `minimize()`take care of both computing the gradients and applying them to the variables: compute the gradients with `compute_gradients()`; process the gradients as you wish; apply the processed gradients with `apply_gradients()`