import tensorflow as tf
import numpy as np
import time

'''
最近看了一下rnn.py源码。由于水平有限，没有完全看懂，看到rnn函数和dynamic_rnn函数时
总感觉这两函数没啥区别，只是一个输入是 list of tensor，另一个是tensor。
且dynamic并没有想象的那么dynamic，只是将填充的部分输出为0
'''
x = tf.placeholder(tf.int32, [None, 1000])
embedding = tf.get_variable('embedding', shape=[100, 256])
x_embedding = tf.nn.embedding_lookup(embedding, x)
'''
embedding_lookup(params, ids)其实就是按照ids顺序返回params中的第ids行。
比如说，ids=[1,3,2],就是返回params中第1,3,2行。返回结果为由params的1,3,2行组成的tensor.
'''
source_sentence_length = tf.placeholder(tf.int32, [None])
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=256)
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, x_embedding, sequence_length=source_sentence_length,
                                                   dtype=tf.float32)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    X_batch = np.random.randint(0, 100, size=[512, 1000])
    time0 = time.time()
    for i in range(100):
        encoder_outputs.eval(feed_dict={x: X_batch, source_sentence_length: [10] * 512})
    time1 = time.time()
    print('sequence_length_10, time: %.9f' % (time1 - time0))
    time2 = time.time()
    for i in range(100):
        encoder_outputs.eval(feed_dict={x: X_batch, source_sentence_length: [1000] * 512})
    time3 = time.time()
    print('sequence_length_1000, time: %.9f' % (time3 - time2))
    '''
    sequence_length_10, time: 695.384773731
    sequence_length_1000, time: 1770.384260178
    '''

'''
具体实现上，调用static_rnn是生成了rnn按时间序列展开之后的图。
打开tensorboard会看到sequence_length个rnn_cell stack在一起，只不过这些cell是share weight(共享权值/参数)的。
因此，sequence_length就和图的拓扑结构绑定在了一起，因此也就限制了每个batch的sequence_length必须是一致。
调用dynamic_rnn不会将rnn展开，而是利用tf.while_loop这个api，
通过Enter, Switch, Merge, LoopCondition, NextIteration等这些control flow的节点，
生成一个可执行循环的图（这个图应该还是静态图，因为图的拓扑结构在执行时是不会变化的）。
在tensorboard上，只会看到一个rnn_cell, 外面被一群control flow节点包围着。
对于dynamic_rnn来说，sequence_length仅仅代表着循环的次数，而和图本身的拓扑没有关系，
所以每个batch可有不同sequence_length。

tf.nn.rnn创建一个展开图的一个固定的网络长度。
这意味着，如果你的电话有200次输入的步骤你与200步骤创建一个静态的图tf.nn.rnn RNN。
首先，图形创建是缓慢的。其次，你不能在较长的序列通过（> 200）比原来specified.tf.nn.dynamic_rnn解决这。
当它被执行时，它使用循环来动态构建图形。这意味着图形创建速度更快，并且可以提供可变大小的批处理。


" 但同一时刻一个batch内部的所有数据长度仍然是固定的。
例如，第一时刻传入的数据shape=[batch_size, 10]，第二时刻传入的数据shape=[batch_size, 12]，
第三时刻传入的数据shape=[batch_size, 8]等等 "这句话看糊涂了。
参考RNNs in Tensorflow, a Practical Guide and Undocumented Features 会更清楚些。
我理解不论dynamic_rnn还是static_rnn（1.0版本没有tf.nn.rnn了），
每个batch的序列长度都是一样的（不足的话自己要去padding），
不同的是dynamic会根据 sequence_length 中止计算。另外一个不同是dynamic_rnn动态生成graph

'''