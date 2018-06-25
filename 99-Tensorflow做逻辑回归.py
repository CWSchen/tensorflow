import tensorflow as tf

# 载入 MINST 数据
from tensorflow.examples.tutorials.mnist import input_data
'''
AttributeError: module ‘pandas’ has no attribute ‘computation’

该问题的原因是pandas中没有computation这个属性，可能是在pandas-0.20.2里没有这个属性吧，
所以需要把pandas的版本换为老版本的（此处以换为pandas-0.19.2为例），而对于此问题的解决步骤如下：

terminal 输入命令 conda update pandas 更新。如果只更新到 0.18.1 那就用下边方法，下载指定版本进行安装
或 从 https://pypi.python.org/pypi/pandas/0.19.2/#downloads 进入网站下载对应版本的pandas，
用的是amd64的Windows版python3.5，点击即可下载：
2、下载完成后，与 terminal 放置在同一个文件夹里，例如：放在 0-python 文件夹里。
pip install pandas-0.19.2-cp35-cp35m-win_amd64.whl  安装即可。
或直接 按照网站提示  pip install pandas==0.19.2 安装。
'''
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
'''
Tensorflow softmax多分类示例
使用MNIST手写数字图像数据作为训练数据，尝试使用softmax做多分类
'''
# 超参数设定
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf 计算图输入
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# 设定模型权重
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 构建softmax模型
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# 计算交叉熵损失
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# 梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初始化向量
init = tf.initialize_all_variables()

# 在会话中启动计算图
with tf.Session() as sess:
    sess.run(init)

    # 多次迭代和更新
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # 多个batches循环
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 用1个batch的数据去fit模型
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # 计算平均损失
            avg_cost += c / total_batch
        # 输出日志
        if (epoch+1) % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print ("Optimization Finished!")

    # 测试一下模型
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算3000个样本上的准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))

