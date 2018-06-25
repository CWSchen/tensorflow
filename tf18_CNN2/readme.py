"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
'''
卷积步长strides参数的具体解释
https://blog.csdn.net/deeplearningfeng/article/details/78551071

conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')

这是一个常见的卷积操作，其中strides=【1,1,1,1】表示滑动步长为1，padding=‘SAME’表示填0操作

当我们要设置步长为2时，strides=【1,2,2,1】，很多同学可能不理解了，这四个参数分别代表了什么，
查了官方函数说明一样不明不白，今天我来解释一下。

strides在官方定义中是一个一维具有四个元素的张量，其规定前后必须为1，这点大家就别纠结了，
所以我们可以改的是中间两个数，中间两个数分别代表了水平滑动和垂直滑动步长值，于是就很好理解了。

在卷积核移动逐渐扫描整体图时候，因为步长的设置问题，可能导致剩下未扫描的空间不足以提供给卷积核的，
大小扫描 比如有图大小为5*5,卷积核为2*2,步长为2,卷积核扫描了两次后，剩下一个元素，
不够卷积核扫描了，这个时候就在后面补零，补完后满足卷积核的扫描，这种方式就是same。
如果说把刚才不足以扫描的元素位置抛弃掉，就是valid方式。
'''