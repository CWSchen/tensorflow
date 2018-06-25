import tensorflow as tf
'''
https://blog.csdn.net/huahuazhu/article/details/73549392
'''

#输入的图像矩阵
input = tf.constant([[1, 1, 1, 0,0],
  [0,1,1,1,0],
  [0,0,1,1,1],
  [0,0,1,1,0],
  [0,1,1,0,0]],shape=[1,5,5,1],dtype=tf.float32)
#卷积核矩阵
filter = tf.constant([[1,0,1],
  [0,1,0],
 [1,0,1]],shape=[3,3,1,1],dtype=tf.float32)
op1 = tf.nn.conv2d(input,filter,strides = [1,1,1,1],padding ='VALID')
#卷积计算
op2 = tf.nn.conv2d(input,filter,strides = [1,1,1,1],padding = 'SAME')
with tf.Session() as sess:
    result1 = sess.run(op1)
    result2 = sess.run(op2)
    print(result1)
    print('#'*100)
    print(result2)



print('#'*100)
#输入的图像矩阵：下面每行数据对应图像矩阵的行；每行中的每组值，分别对应0,1,2三个通道相应位置的值
input = tf.constant([[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
 [[0,0,0],[0,0,1],[1,0,0],[2,1,1],[0,0,2],[1,0,2],[0,0,0]],
 [[0,0,0],[2,0,0],[0,1,1],[1,1,2],[0,2,0],[0,1,1],[0,0,0]],
 [[0,0,0],[0,0,0],[0,2,1],[2,1,0],[2,0,2],[1,0,2],[0,0,0]],
 [[0,0,0],[2,2,0],[2,1,1],[0,2,1],[1,1,2],[0,0,0],[0,0,0]],
 [[0,0,0],[1,2,2],[2,1,2],[0,1,2],[0,1,1],[0,2,1],[0,0,0]],
 [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]],shape=[1,7,7,3],dtype=tf.float32)
#input = tf.truncated_normal([1,5,5,3],stddev=0.1)
#卷积核矩阵1
#filter = tf.truncated_normal([3,3,3,2], stddev=0.1)
bias = tf.constant([1,0],shape=[2],dtype=tf.float32)
#滤波器（卷积核）矩阵：两个卷积核，每个卷积核有三个通道，共六个矩阵。第一行[[-1,1],[0,0],[0,1]]为六个矩阵第0行0列的值，[-1,1]第一个数-1是第一个卷积核W0第0通道中（也就是第一个矩阵）第0行第0列的值，第二个数1是第二个卷积核W1第0通道中第0行第0列的值；[0,0] 第一个数0是第一个卷积核W0第1通道中第0行第0列的值，第二个数0是第二个卷积核W1第1通道中第0行第0列的值；以此类推。
filter = tf.constant([[[[-1,1],[0,0],[0,1]],
  [[0,1],[-1,-1],[-1,0]],
  [[1,0],[-1,0],[1,0]]],
 [[[0,-1],[-1,-1],[0,-1]],
  [[1,-1],[1,0],[0,-1]],
  [[0,0],[-1,-1],[-1,0]]],
 [[[0,1],[0,-1],[0,1]],
  [[-1,1],[-1,-1],[1,-1]],
  [[-1,-1],[1,1],[-1,-1]]]],  shape=[3,3,3,2],dtype=tf.float32)
#卷积计算，两个维度窗口滑动步长分别都为2
op1 = tf.nn.conv2d(input,filter,strides = [1,2,2,1],padding ='VALID')+bias
op2 = tf.nn.conv2d(input,filter,strides = [1,2,2,1],padding = 'SAME')+bias
with tf.Session() as sess:
    result1 = sess.run(op1)
    result2 = sess.run(op2)
    print(sess.run(input))
    print(sess.run(filter))
    print(result1)