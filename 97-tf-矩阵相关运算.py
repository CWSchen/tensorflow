import tensorflow as tf
import numpy as np
'''
TensorFlow 将图形定义转换成分布式执行的操作, 以充分利用可用的计算资源
(如 CPU 或 GPU。一般你不需要显式指定使用 CPU 还是 GPU, TensorFlow 能自动检测。
如果检测到 GPU, TensorFlow 会尽可能地利用找到的第一个 GPU 来执行操作.
并行计算能让代价大的算法计算加速执行，TensorFlow也在实现上对复杂操作进行了有效的改进。
大部分核相关的操作都是设备相关的实现，比如GPU。下面是一些重要的操作/核：
'''

'''
返回一个给定对角值的对角tensor
'''
w = [1, 2, 3, 4]
op1 = tf.diag(w)

'''需要依赖前面的 op的结果值'''
def def_op2(result):
    op2 = tf.diag_part(result)
    return op2

w2 = np.arange(0,12).reshape(3,4)
# print(w2)
'''
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
'''
op3 = tf.trace(w2)
op4 = tf.transpose(w2)

# 启动图, 运行 op
with tf.Session() as sess:
    result = sess.run(op1)
    print('tf.diag 返回一个给定对角值的对角tensor \n',result)
    op2 = def_op2(result)
    result2 = sess.run(op2)
    print('tf.diag_part 与 tf.diag 功能相反 \n',result2)
    result3 = sess.run(op3)
    print('tf.trace(x, name=None)求一个2维tensor足迹，即对角值diagonal之和 \n',result3)
    result4 = sess.run(op4)
    print('tf.transpose(x)调换tensor的维度顺序，就是转置 \n',result4)

