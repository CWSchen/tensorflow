import tensorflow as tf
'''
https://blog.csdn.net/huahuazhu/article/details/73649389
一、 Dropout原理简述：
tf.nn.dropout是TensorFlow里面为了防止或减轻过拟合而使用的函数，它一般用在全连接层。
Dropout就是在不同的训练过程中随机扔掉一部分神经元。也就是让某个神经元的激活值以一定的概率p，
让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。
但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了。示意图如下：
但在测试及验证中：每个神经元都要参加运算，但其输出要乘以概率p。

二、tf.nn.dropout函数说明
tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None,name=None)
上面方法中常用的是前两个参数：
第一个参数x：指输入
第二个参数keep_prob: 设置神经元被选中的概率,在初始化时keep_prob是一个占位符,
keep_prob = tf.placeholder(tf.float32) 。tensorflow在run时设置keep_prob具体的值，例如keep_prob: 0.5
第五个参数name：指定该操作的名字。

三、使用举例：
1、dropout必须设置概率keep_prob,并且keep_prob也是一个占位符,跟输入是一样的

[python] view plain copy

1. keep_prob = tf.placeholder(tf.float32)

2、train的时候才是dropout起作用的时候,train和test的时候不应该让dropout起作用

[python] view plain copy

1. sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})

[python] view plain copy

1. train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})

2. test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
'''

'''
https://blog.csdn.net/Eclipsesy/article/details/77603336
'''
a = tf.constant([[1.,2.],[5.,-2.]])
relu_a = tf.nn.relu(a)
sigmoid_a = tf.nn.sigmoid(a)
tanh_a = tf.nn.tanh(a)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result_relu_a = sess.run(relu_a)
    result_sigmoid_a = sess.run(sigmoid_a)
    result_tanh_a = sess.run(tanh_a)
    print('the result of relu(a) is : \n{}'.format(result_relu_a))
    print('the result of sigmoid(a) is : \n{}'.format(result_sigmoid_a))
    print('the result of tanh(a) is : \n{}'.format(result_tanh_a))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    d = tf.constant([[1.,2.,3.,4.],[5.,6.,7.,8.],[9.,10.,11.,12.],[13.,14.,15.,16.]])
    print(sess.run(tf.shape(d)))

    #由于[4,4] == [4,4] 行和列都为独立
    dropout_a44 = tf.nn.dropout(d, 0.5, noise_shape = [4,4])
    result_dropout_a44 = sess.run(dropout_a44)
    print(result_dropout_a44)

    #noise_shpae[0]=4 == tf.shape(d)[0]=4
    #noise_shpae[1]=4 != tf.shape(d)[1]=1
    #所以[0]即行独立，[1]即列相关，每个行同为0或同不为0
    dropout_a41 = tf.nn.dropout(d, 0.5, noise_shape = [4,1])
    result_dropout_a41 = sess.run(dropout_a41)
    print(result_dropout_a41)

    #noise_shpae[0]=1 ！= tf.shape(d)[0]=4
    #noise_shpae[1]=4 == tf.shape(d)[1]=4
    #所以[1]即列独立，[0]即行相关，每个列同为0或同不为0
    dropout_a24 = tf.nn.dropout(d, 0.5, noise_shape = [1,4])
    result_dropout_a24 = sess.run(dropout_a24)
    print(result_dropout_a24)
    #不相等的noise_shape只能为1
