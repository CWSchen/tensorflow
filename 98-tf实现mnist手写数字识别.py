import tensorflow as tf
'''
https://blog.csdn.net/huahuazhu/article/details/74199580

卷积神经网络CNN的理论部分直接看帖子：TensorFlow学习--卷积神经网络CNN
卷积神经网络CNN的结构一般包含这几个层：
输入层：用于数据的输入
卷积层：使用卷积核进行特征提取和特征映射
激活层：由于卷积也是一种线性运算，因此需要增加非线性映射。激活函数，并不是去激活什么，
    而是指如何把“激活的神经元的特征”通过函数把特征保留并映射出来（保留特征，去除一些数据中是的冗余），
    这是神经网络能解决非线性问题关键。在卷积神经网络中，对于图像，我们主要采用了卷积的方式来处理，
    也就是对每个像素点赋予一个权值，这个操作显然就是线性的。但是对于我们样本来说，
    不一定是线性可分的，为了解决这个问题，我们引入非线性因素，也就是通过激活函数解决线性模型所不能解决的问题。
池化层：进行下采样，对特征图稀疏处理，减少数据运算量。
全连接层：通常在CNN的尾部进行重新拟合，减少特征信息的损失。卷积神经网络特征提取完成后，这时的特征维度相对于原始的图片输入已经大大减小，这时可以采用全连接层来最大限度利用每一个特征。全连接层的每一个节点都与上一层的所有节点相连，用来把前边提取到的特征综合起来。
Dropout层：为了防止过拟合而使用的。
输出层：用于输出结果
'''
import sys
'''
这都是 python 2 的，python 3 里 没有
# reload(sys)
# sys.setdefaultencoding('utf8')
'''
  
import tensorflow as tf   
from tensorflow.examples.tutorials.mnist import input_data   
    
mnist = input_data.read_data_sets('/root/tftest/data/', one_hot=True)   
'''
#这里使用TensorFlow实现一个简单的卷积神经网络，使用的是MNIST数据集。
# 网络结构为：数据输入层–卷积层1–池化层1–卷积层2–池化层2–全连接层1–全连接层2（输出层），
# 这是一个简单但非常有代表性的卷积神经网络。
'''
#28*28的灰度图，像素个数784  
#是10分类问题
x = tf.placeholder("float", shape=[None, 784])   
y_ = tf.placeholder("float", shape=[None, 10])   
  
#初始化单个卷积核上的参数  
#使用截断的正太分布，标准差为 0.1 的初始值来初始化权值  
def weight_variable(shape):  
     initial = tf.truncated_normal(shape, stddev=0.1)  
     return tf.Variable(initial)  
  
#初始化单个卷积核上的偏置值  
#使用常量 0.1 来初始化偏置 B  
def bias_variable(shape):  
     initial = tf.constant(0.1, shape=shape)  
     return tf.Variable(initial)  
  
#输入特征x，用卷积核W进行卷积运算，strides为卷积核移动步长、多少个channel、filter的个数即产生特征图的个数  
#padding表示是否需要补齐边缘像素使输出图像大小不变  
def conv2d(x, W):  
     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  
  
#对x进行最大池化操作，ksize进行池化的范围  
#使用步长为2，下采样核为2 * 2 的方式来初始化池化层  
# 2x2最大池化，即将一个2x2的像素块降为1x1的像素。最大池化会保留原始像素块中灰度值最高的那一个像素，即保留最显著的特征。  
def max_pool_2x2(x):  
     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],  
                           strides=[1, 2, 2, 1], padding='SAME')  
  
#先将mnist数据集图片还原为二维向量结构，28 * 28 = 784  1代表颜色通道数量,-1表示自动计算此维度  
x_image = tf.reshape(x, [-1,28,28,1])  
  
#第一个卷积层, 总共 32 个 5 * 5 的卷积核，由于使用 'SAME' 填充，因此卷积后的图片尺寸依然是 28 * 28  
#把x_image的厚度由1增加到32，长宽由28*28缩小为14*14  
W_conv1 = weight_variable([5, 5, 1, 32])  # 按照[5,5,输入通道=1,输出通道=32]生成一组随机变量  
b_conv1 = bias_variable([32])  
#进行卷积操作，并添加relu激活函数  
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 输出size 28*28*32(因为conv2d()中x和y步长都为1，边距保持不变)  
#第一个池化层，28 * 28的图片尺寸池化后，变为 14 * 14  
h_pool1 = max_pool_2x2(h_conv1)   # 输出size 14*14*32  
  
#第二个卷积层, 总共 64 个 5 * 5 的卷积核，由于使用 'SAME' 填充，因此卷积后的图片尺寸依然是 14 * 14  
#把h_pool1的厚度由32增加到64，长宽由14*14缩小为7*7  
W_conv2 = weight_variable([5, 5, 32, 64])   #64个厚度32的 5 * 5 卷积核  
b_conv2 = bias_variable([64])  
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  
#第二个池化层，14 * 14 的图片尺寸池化后，变为 7 * 7  
h_pool2 = max_pool_2x2(h_conv2)   # 输出size 7*7*64  
  
#第一层全连接,把h_pool2由7*7*64，变成1024*1  
#使用上面定义的方式初始化接下来的两个全连接层的参数 W 和 B  
# fc1，将两次池化后的7*7共128个特征图转换为1D向量，隐含节点1024由自己定义  
W_fc1 = weight_variable([7 * 7 * 64, 1024])  
b_fc1 = bias_variable([1024])  
#将卷积的产出展开,将二维图片结构转化为一维图片结构  
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  
#神经网络计算，并添加relu激活函数.第一个全连接层，图片尺寸从 7 * 7 * 64维变换为1024维  
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  
#为了减轻过拟合，使用Dropout层  
keep_prob = tf.placeholder("float")  
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  
  
#第二层全连接,输出层，使用softmax计算概率进行多分类,最后一层网络,从 1024 维变换为 10 维的one-hot向量  
W_fc2 = weight_variable([1024, 10])  
b_fc2 = bias_variable([10])  
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)    
  
#计算交叉熵loss，并使用自适应动量梯度下降算法优化loss  
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))   
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)   
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)   
  
#计算准确率  
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))   
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))   
  
#定义一个交互式的session，并初始化所有变量  
sess = tf.InteractiveSession()  
sess.run(tf.initialize_all_variables())   
#print((tf.shape(W_conv2)[0])  
for i in range(0,5):  
    for j in range(0,5):  
        print(((W_conv2.eval()[i][j][31][63]))  )
saver=tf.train.Saver(max_to_keep=1)  
for i in range(1000):   
     batch = mnist.train.next_batch(50)   
     train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})   
     if i%100 == 0:   
         train_accuracy = accuracy.eval(feed_dict={   
             x:batch[0], y_: batch[1], keep_prob: 1.0})   
         print( "step %d, training accuracy %g"%(i, train_accuracy)   )
  
#saver.save(sess, "/root/tftest/model_data/mnist_cnn.ckpt", global_step=i+1)  
    
print( "test accuracy %g"
       %accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}) )
  
for i in range(0,5):  
    for j in range(0,5):  
        print(((W_conv2.eval()[i][j][31][63]))  )

'''
若要使用自己的图片数据进行训练，可以将上面程序修改如下地方，将读入图片的语句，改为如下

for i in range(1000):

    #batch = mnist.train.next_batch(1)

     batch= next_batch(50, i)

，其中next_batch()为自己读入图片的代码

root = '/root/tftest/data/'

f = open(root + 'mnist/train/train.txt')

#fpath = []

#num = []

fpath = [line.strip().split(" ")[0]for line in f]

f.close

f = open(root + 'mnist/train/train.txt')

num = [int(line.strip().split(" ")[1])for line in f]

f.close



f = open(root + 'mnist/test/test.txt')

#fpath = []

#num = []

fpath_test = [line.strip().split("")[0] for line in f]

f.close

f = open(root + 'mnist/test/test.txt')

num_test = [int(line.strip().split("")[1]) for line in f]

f.close



def next_batch(batch_size, step):

   #global fpath

   #global num

    bath0= np.zeros( (batch_size, 784) )

    bath1= np.zeros( (batch_size, 10) )

    j=0

    for iin range(batch_size * step, batch_size * step + batch_size):

       #print(root + fpath[i])

        im= cv2.imread(root + fpath[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)

        im= cv2.resize(im,(28,28),interpolation=cv2.INTER_CUBIC)

        #img_gray = 255.0 - im

       img_gray = (255.0 - im) / 255.0

       x_img = np.reshape(img_gray , [-1 , 784])

       bath0[j] = x_img

       #print(num[i])

       bath1[j][num[i]] = 1.0

        j+= 1

    returnbath0, bath1
'''