import tensorflow as tf
from pprint import pprint
'''
tensorflow - tf.nn.conv2d实现卷积的原理
参考：http://www.cnblogs.com/welhzh/p/6607581.html

tf.nn.conv2d 是 tf 里面实现卷积的函数，是搭建卷积神经网络比较核心的方法，非常重要。
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)

除去name参数，与方法有关的一共五个参数：
第一个参数 input：指需要做卷积的输入图像，
要求是一个 Tensor [batch, in_height, in_width, in_channels]，类型为float32和float64其中之一
具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，是一个4维的Tensor。

第二个参数 filter：相当于CNN中的卷积核，
要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，
有一个地方需要注意，第三维in_channels，就是参数input的第四维

第三个参数strides：卷积时针对图像每一维的步长，这是一个一维的向量，长度4

第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）
    ## valid 有效的,正当的,指仍为权威部门认可

第五个参数：use_cudnn_on_gpu，bool类型，是否使用 cudnn 加速，默认为true
   # NVIDIA CuDNN 是专门针对DeepLearning框架设计的一套GPU计算加速方案，
   目前支持的DL库包括Caffe，ConvNet, Torch7等。

   # CUDA(Compute Unified Device Architecture 统一计算设备架构)，是显卡厂商NVIDIA推出的运算平台。
   CUDA是一种由NVIDIA推出的通用并行计算架构，该架构使GPU能够解决复杂的计算问题。
   它包含了CUDA 指令集架构(Instruction Set Architecture) 以及GPU内部的并行计算引擎。

结果返回一个Tensor，这个输出，就是我们常说的 feature map
'''
'''
TensorFlow 的卷积具体是怎样实现的呢？举个栗子解释它：

case 1 考虑一种最简单的情况，现在有一张 3×3 单单通道的图像（对应的shape：[1，3，3，1]），
用一个1×1的卷积核（对应的shape：[1，1，1，1]）去做卷积，最后会得到一张 3×3 的 feature map
'''
input = tf.Variable(tf.random_normal([1,3,3,1]))
filter = tf.Variable(tf.random_normal([1,1,1,1]))

op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
'''
case 2 增加图片的通道数，使用一张 3×3 五通道的图像（对应的shape：[1，3，3，5]），
用一个1×1的卷积核（对应的shape：[1，1，1，1]）去做卷积，仍然是一张3×3的feature map，
这就相当于每一个像素点，卷积核都与该像素点的每一个通道做点积
'''
input = tf.Variable(tf.random_normal([1,3,3,5]))
filter = tf.Variable(tf.random_normal([1,1,5,1]))

op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

'''
#case 3 把卷积核扩大，现在用3×3的卷积核做卷积，
最后的输出是一个值，相当于 case 2 的feature map所有像素点的值求和
'''
input = tf.Variable(tf.random_normal([1,3,3,5]))
filter = tf.Variable(tf.random_normal([3,3,5,1]))

op3 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

'''
#case 4 使用更大的图片将情况2的图片扩大到5×5，仍然是3×3的卷积核，令步长为1，输出3×3的feature map
'''
input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,1]))

op4 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

'''
注意我们可以把这种情况看成情况2和情况3的中间状态，卷积核以步长1滑动遍历全图，
以下x表示的位置，表示卷积核停留的位置，每停留一个，输出 feature map 的一个像素

.....
.xxx.
.xxx.
.xxx.
.....

case 5 上面我们一直令参数 padding的值为‘VALID’，
当其为‘SAME’时，表示卷积核可以停留在图像边缘，如下，输出5×5的feature map
'''
input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,1]))

op5 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

'''
xxxxx
xxxxx
xxxxx
xxxxx

#case 6 如果卷积核有多个
'''
input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,7]))

op6 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

'''
此时输出7张5×5的feature map

case 7 步长不为1的情况，文档里说了对于图片，因为只有两维，通常strides取[1，stride，stride，1]
'''
input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,7]))

op7 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')

'''
此时，输出7张3×3的feature map

x.x.x
.....
x.x.x
.....
x.x.x

case 8 如果batch值不为1，同时输入10张图
'''
input = tf.Variable(tf.random_normal([10,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,7]))

op8 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')

'''
每张图，都有7张3×3的feature map，输出的shape就是[10，3，3，7]
'''
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print("case 1 \n",sess.run(op))
    print("case 2 \n",sess.run(op2))
    print("case 3 \n",sess.run(op3))
    print("case 4 \n",sess.run(op4))
    print("case 5 \n",sess.run(op5))
    print("case 6 \n",sess.run(op6))
    print("case 7 \n",sess.run(op7))
    print("case 8 \n",sess.run(op8))

    # print("case 2")
    # pprint(sess.run(op2))
    # print("case 3")
    # pprint(sess.run(op3))
    # print("case 4")
    # pprint(sess.run(op4))
    # print("case 5")
    # pprint(sess.run(op5))
    # print("case 6")
    # pprint(sess.run(op6))
    # print("case 7")
    # pprint(sess.run(op7))
    # print("case 8")
    # pprint(sess.run(op8))