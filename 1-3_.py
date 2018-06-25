import tensorflow as tf
import numpy as np

'''
tensorflow中文社区：http://www.tensorfly.cn/tfdoc/get_started/basic_usage.html

MYPS：学习它，需要很多基础知识，建议大家买视频课程一步步做笔记积累的学习，
如果第一轮有看不懂的，不明白的，就先越过，
因为不是老师讲的不清楚，而是自己的知识储备不够，听不明白。
然后，我们就开始了迭代.....这个过程很难熬，即使看不懂也要看，坚持读下去，
慢慢的，就会发现，之前画问号的地方，都不用再花费时间看视频了，自己就懂了
培训机构的课程，是仅可以让我们入门的。然后能走多远，多深入，就看自己的努力了！

tensorflow入门教程和底层机制简单解说——本质就是图计算，自动寻找依赖，想想spark机制就明白了
简介，本章的目的是让你了解和运行 TensorFlow!
在开始之前, 先看一段使用 Python API 撰写的 TensorFlow 示例代码, 对将要学习的内容有初步的印象.
这段很短的 Python 程序生成了一些三维数据, 然后用一个平面拟合它.

一，tensorflow中有一类在tensor的某一维度上求值的函数。如：
求最大值tf.reduce_max(input_tensor, reduction_indices=None, keep_dims=False, name=None)
求平均值tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)

参数1--input_tensor:待求值的tensor。
参数2--reduction_indices:在哪一 维(列) 上求解。
参数（3）（4）可忽略
举例说明：
# 'x' is [[1., 2.]
#         [3., 4.]]
x是一个2维数组，分别调用reduce_*函数如下：
首先求平均值：
tf.reduce_mean(x) ==> 2.5 #如果不指定第二个参数，那么就在所有的元素中取平均值
tf.reduce_mean(x, 0) ==> [2.,  3.] #指定第二个参数为0，则第一维的元素取平均值，即每一列求平均值
tf.reduce_mean(x, 1) ==> [1.5,  3.5] #指定第二个参数为1，则第二维的元素取平均值，即每一行求平均值
同理，还可用tf.reduce_max()求最大值等。

tf.multiply（）和tf.matmul（）区别
（1）tf.multiply是点乘，逐位/对应位置 元素相乘，即 Returns x * y element-wise.
（2）tf.matmul是矩阵乘法，即 Multiplies matrix a by matrix b, producing a * b.
'''

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100))  # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 构造一个线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]

'''
疑问：底层如何运作？

难道自动感知下面这些变量？怎么知道是要做模型的训练和优化？我又没有告知它使用什么模型？

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
见下面文档，本质上就是图计算，run（xxx）的时候会自动去寻找变量依赖！！！

最能表达tensorflow底层机制的例子是：
'''
# 创建一个变量, 初始化为标量 0.
state = tf.Variable(0, name="counter")
# 创建一个 op, 其作用是使 state 增加 1
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.
init_op = tf.initialize_all_variables()

# 启动图, 运行 op
with tf.Session() as sess:
    # 运行 'init' op
    sess.run(init_op)
    # 打印 'state' 的初始值
    print(sess.run(state))
    # 运行 op, 更新 'state', 并打印 'state'
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

'''
基本使用
使用 TensorFlow, 必须明白 TensorFlow:使用图 (graph) 来表示计算任务.
在被称之为 会话 (Session) 的上下文 (context) 中执行图.
使用 tensor 表示数据.通过 变量 (Variable) 维护状态.
使用 feed(馈送,喂养,进给) 和 fetch(取得) 可以为任意的操作(arbitrary operation) 赋值或者从其中获取数据.

综述，TensorFlow 是一个编程系统, 使用图来表示计算任务. 图中的节点被称之为 op (operation 的缩写).
一个 op 获得 0 个或多个 Tensor, 执行计算, 产生 0 个或多个 Tensor. 每个 Tensor 是一个类型化的多维数组.
例如, 你可以将一小组图像集表示为一个四维浮点数数组, 这四个维度分别是 [batch(分批处理), height, width, channels(通道)].

一个 TensorFlow 图描述了计算的过程. 为了进行计算, 图必须在 会话 里被启动. 会话 将图的 op 分发到诸如 CPU 或 GPU 之类的 设备 上,
同时提供执行 op 的方法. 这些方法执行后, 将产生的 tensor 返回. 在 Python 语言中, 返回的 tensor 是 numpy ndarray 对象;
在 C 和 C++ 语言中, 返回的 tensor 是 tensorflow::Tensor 实例.

计算图 TensorFlow 程序通常被组织成一个构建阶段和一个执行阶段.
在构建阶段, op 的执行步骤 被描述成一个图. 在执行阶段, 使用会话执行执行图中的 op.

例如, 通常在构建阶段创建一个图来表示和训练神经网络, 然后在执行阶段反复执行图中的训练 op.

TensorFlow 支持 C, C++, Python 编程语言.
目前, TensorFlow 的 Python 库更加易用, 提供了大量的辅助函数来简化构建图的工作, 这些函数尚未被 C 和 C++ 库支持.
三种语言的会话库 (session libraries) 是一致的.

## 构建图的第一步, 是创建源 op (source op). 源 op 不需要任何输入, 例如 常量 (Constant). 源 op 的输出被传递给其它 op 做运算.

Python 库中, op 构造器的返回值代表被构造出的 op 的输出, 这些返回值可以传递给其它 op 构造器作为输入.

TensorFlow Python 库有一个默认图 (default graph), op 构造器可为其增加节点. 这个默认图对许多程序来说已经足够用了.
阅读 Graph 类 文档 来了解如何管理多个图.

# 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点
# 加到默认图中.
#
# 构造器的返回值代表该常量 op 的返回值.
matrix1 = tf.constant([[3., 3.]])

# 创建另外一个常量 op, 产生一个 2x1 矩阵.
matrix2 = tf.constant([[2.],[2.]])

# 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
# 返回值 'product' 代表矩阵乘法的结果.
product = tf.matmul(matrix1, matrix2)
默认图现在有三个节点, 两个 constant() op, 和一个matmul() op. 为了真正进行矩阵相乘运算, 并得到矩阵乘法的 结果, 你必须在会话里启动这个图.

在一个会话中启动图，构造阶段完成后, 才能启动图.
启动图的第一步是创建一个 Session 对象, 如果无任何创建参数, 会话构造器将启动默认图.

欲了解完整的会话 API, 请阅读Session 类.

# 启动默认图.
sess = tf.Session()

# 调用 sess 的 'run()' 方法来执行矩阵乘法 op, 传入 'product' 作为该方法的参数.
# 上面提到, 'product' 代表了矩阵乘法 op 的输出, 传入它是向方法表明, 我们希望取回
# 矩阵乘法 op 的输出.
#
# 整个执行过程是自动化的, 会话负责传递 op 所需的全部输入. op 通常是并发执行的.
#
# 函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行.
#
# 返回值 'result' 是一个 numpy `ndarray` 对象.
result = sess.run(product)
print result
# ==> [[ 12.]]

# 任务完成, 关闭会话.
sess.close()
Session 对象在使用完后需要关闭以释放资源. 除了显式调用 close 外, 也可以使用 "with" 代码块 来自动完成关闭动作.

with tf.Session() as sess:
  result = sess.run([product])
  print result
在实现上, TensorFlow 将图形定义转换成分布式执行的操作, 以充分利用可用的计算资源(如 CPU 或 GPU).
一般不需要显式指定使用 CPU 还是 GPU, TensorFlow 能自动检测. 如果检测到 GPU, TensorFlow 会尽可能地利用找到的第一个 GPU 来执行操作.

如果机器上有超过一个可用的 GPU, 除第一个外的其它 GPU 默认是不参与计算的. 为了让 TensorFlow 使用这些 GPU,
必须将 op 明确指派给它们执行. with...Device 语句用来指派特定的 CPU 或 GPU 执行操作:

with tf.Session() as sess:
  with tf.device("/gpu:1"):
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.],[2.]])
    product = tf.matmul(matrix1, matrix2)
    ...
设备用字符串进行标识. 目前支持的设备包括:

"/cpu:0": 机器的 CPU.
"/gpu:0": 机器的第一个 GPU, 如果有的话.
"/gpu:1": 机器的第二个 GPU, 以此类推.

阅读使用GPU章节, 了解 TensorFlow GPU 使用的更多信息.

交互式使用
文档中的 Python 示例使用一个会话 Session 来 启动图, 并调用 Session.run() 方法执行操作.

为了便于使用诸如 IPython 之类的 Python 交互环境, 可以使用 InteractiveSession 代替 Session 类,
使用 Tensor.eval() 和 Operation.run() 方法代替 Session.run(). 这样可以避免使用一个变量来持有会话.

# 进入一个 Interactive (交互式,互动,相互作用的) 交互式 TensorFlow 会话.
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# 使用初始化器 initializer op 的 run() 方法初始化 'x'
x.initializer.run()

# 增加一个减法 sub op, 从 'x' 减去 'a'. 运行减法 op, 输出结果
sub = tf.sub(x, a)
print sub.eval()
# ==> [-2. -1.]

##Tensor
TensorFlow 程序使用 tensor 数据结构来代表所有的数据, 计算图中, 操作间传递的数据都是 tensor.
可把 TensorFlow tensor 看作是一个 n 维的数组或列表. 一个 tensor 包含一个静态类型 rank(队列), 和 一个 shape(形状,维度).
想了解 TensorFlow 是如何处理这些概念的, 参见 Rank, Shape, 和 Type.

##变量
Variables for more details. 变量维护图执行过程中的状态信息（例如最开始wx+b的拟合，w和b就是计算状态）.
下面的例子演示了如何使用变量实现一个简单的计数器. 参见 变量 章节了解更多细节.

加深记忆，再次重温！！！ 不要嫌基础烦。好的基础，才决定上层建筑！
'''
# 创建一个变量, 初始化为标量 0.
state = tf.Variable(0, name="counter")

# 创建一个 op, 其作用是使 state 增加 1
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.
init_op = tf.initialize_all_variables()

# 启动图, 运行 op
with tf.Session() as sess:
    # 运行 'init' op
    sess.run(init_op)
    # 打印 'state' 的初始值
    print(sess.run(state))
    # 运行 op, 更新 'state', 并打印 'state'
    for i in range(3):
        sess.run(update)
        print(sess.run(state))

# 输出:
# 0
# 1
# 2
# 3
'''
代码中 assign() 操作是图所描绘的表达式的一部分, 正如 add() 操作一样. 所以在调用 run() 执行表达式之前, 它并不会真正执行赋值操作.

通常会将一个统计模型中的参数表示为一组变量.
例如, 可将一个神经网络的权重作为某个变量存储在一个 tensor 中. 在训练过程中, 通过重复运行训练图, 更新这个 tensor

Fetch(取得)
为了取回操作的输出内容, 可在使用 Session 对象的 run() 调用 执行图时, 传入一些 tensor 会帮助取回结果.
在之前的例子里, 只取回了单个节点 state, 同时也是可以取回多个 tensor:

    input1 = tf.constant(3.0)
    input2 = tf.constant(2.0)
    input3 = tf.constant(5.0)
    intermed = tf.add(input2, input3)
    mul = tf.mul(input1, intermed)

    with tf.Session():
      result = sess.run([mul, intermed])
      print result

    # 输出:
    # [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]
    需要获取的多个 tensor 值，在 op 的一次运行中一起获得（而不是逐个去获取 tensor）。

Feed(供给)
上述示例在计算图中引入了 tensor, 以常量或变量的形式存储. TensorFlow 还提供了 feed 机制,
该机制 可临时替代图中的任意操作中的 tensor 可对图中任何操作提交补丁, 直接插入一个 tensor.

feed 使用一个 tensor 值，临时替换一个操作的输出结果.
可提供 feed 数据作为 run() 调用的参数. feed 只在调用它的方法内有效。方法结束, feed 就会消失.
最常见的用例是将某些特殊的操作指定为 "feed" 操作, 标记的方法是使用 tf.placeholder() 为这些操作创建占位符.

    input1 = tf.placeholder(tf.types.float32)
    input2 = tf.placeholder(tf.types.float32)
    output = tf.mul(input1, input2)

    with tf.Session() as sess:
      print sess.run([output], feed_dict={input1:[7.], input2:[2.]})

    # 输出:
    # [array([ 14.], dtype=float32)]
    for a larger-scale example of feeds. 如果没有正确提供 feed, placeholder() 操作将会产生错误.
    MNIST 全连通 feed 教程 (source code) 给出了一个更大规模的使用 feed 的例子.

'''

