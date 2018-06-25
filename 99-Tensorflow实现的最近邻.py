import numpy as np
import tensorflow as tf

# 载入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
'''
Extracting /tmp/data/train-images-idx3-ubyte.gz
Extracting /tmp/data/train-labels-idx1-ubyte.gz
Extracting /tmp/data/t10k-images-idx3-ubyte.gz
Extracting /tmp/data/t10k-labels-idx1-ubyte.gz
'''
# 防止内存占用太多，只选取一部分数据
Xtr, Ytr = mnist.train.next_batch(5000) #5000 for training (nn candidates)
Xte, Yte = mnist.test.next_batch(200) #200 for testing

# tf 计算图输入
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

# 使用曼哈顿/L1距离
'''
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.neg(xte))), reduction_indices=1) 报错：
AttributeError: module 'tensorflow' has no attribute 'neg'，新版本改为 tf.negative 啦
'''
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# 找到最近邻
pred = tf.arg_min(distance, 0)

accuracy = 0.

# 初始化变量
init = tf.initialize_all_variables()


# 在session中启动图计算
with tf.Session() as sess:
    sess.run(init)

    # 遍历测试集
    for i in range(len(Xte)):
        # 找到最近邻
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        # 输出结果
        print ("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
            "True Class:", np.argmax(Yte[i]))
        # 计算准确率
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print ("Done!")
    print ("Accuracy:", accuracy)