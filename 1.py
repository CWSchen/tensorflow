# coding: utf-8
import tensorflow as tf
import numpy as np
print(tf.__version__)  # 1.6.0
'''
首先，通过tf.constant创建一个常量，然后启动Tensorflow的Session，
调用sess的run方法来启动整个graph。接下来我们做下简单的数学的方法：
'''
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1, matrix2)  # matrix multiply np.dot(m1, m2)

'''
# method 1 - 非 with 方法 需要 close
'''
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()
'''
# method 2 - with 方法，是运行完成后，自动close
'''
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)


x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))  # 计算张量的各个维度上的元素的平均值
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
### create tensorflow structure end ###

sess = tf.Session()
sess.run(init)  # Very important

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))

'''
tf.placeholder(dtype, shape=None, name=None)
此函数可以理解为函数中的形参，用于定义过程，在执行时赋具体的值
参数：
    dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
    shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2,3], [None, 3]表示列是3，行不定
    name：名称
'''

x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

with tf.Session() as sess:
    # print(sess.run(y))  # ERROR: 此处x还没有赋值.

    '''
    feed_dict 传入参数 feeds
    sess.run(y, feed_dict={x: rand_array})
    '''
    rand_array = np.random.rand(1024, 1024)
    print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.

# TensorFlow 还提供了 feed 机制, 可临时替代图中的任意操作中的 tensor ，可对图中任何操作提交补丁, 直接插入一个 tensor.
#
# feed 使用一个 tensor 值临时替换一个操作的输出结果. 可提供 feed 数据作为 run() 调用的参数.
# feed 只在调用它的方法内有效, 方法结束, feed 就会消失。通常将某些特殊的操作指定为 "feed" 操作,
# 标记的方法是使用 tf.placeholder() 为这些操作创建占位符.

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1: [7.], input2: [2.]}))
    '''
    [array([14.], dtype=float32)]
    '''

'''
module 'tensorflow' has no attribute 'xxx'
原因是TensorFlow新版本修改了许多函数的名字，不完全整理如下：
tf.sub()更改为tf.subtract()
tf.mul()更改为tf.multiply()
tf.types.float32更改为tf.float32
tf.pact()更改为tf.stact() # pact  n. 协定；公约；条约；契约
'''

'''
tf.Variable(<variable_name>)会自动检测命名冲突并自行处理,可定义名字相同的变量，
tf.Variable函数会返回一个variable，如果给出的name已经存在，会自动修改name，生成个新的
比如：下面定义两个名字同为a1的变量，但是，系统为自动给重新分配变量名
'''
state = tf.Variable(0, name='counter')
print('state.name相同 = ' , state.name)
state = tf.Variable(1, name='counter')
print('state.name相同 = ' , state.name)
'''
state.name相同 =  counter:0
state.name相同 =  counter_1:0

# 这么写是不对的 AttributeError: 'Variable' object has no attribute 'counter'
print('state.name.counter = ' , state.name.counter)
'''

'''
tf.get_variable(<variable_name>)则遇到重名的变量创建且变量名没有设置为共享变量时，则会报错。
下面的代码中，使用tf.get_variable时，加粗字体处必须不同，否则运行报错；
使用tf.Variable则不会报错，会自动处理添加 _1 和_2 进行区分。
tf.get_variable()不可以定义名字相同的变量，
tf.get_variable函数拥有一个变量检查机制，会检测已经存在的变量是否设置为共享变量，
如果已经存在的变量没有设置为共享变量，TensorFlow 运行到第二个拥有相同名字的变量的时候，就会报错。
不同的变量之间不能有相同的名字，除非你定义了variable_scope，这样才可以有相同的名字。
'''


one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.initialize_all_variables()  # must have if define variable

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        result_ = sess.run(update)
        print('sess.run(state) = ' , result_)

'''
var1 = tf.get_variable(name='var2', shape=[1], dtype=tf.float32 ,  initializer= [1.])
var3 = tf.get_variable(name='var2', shape=[1], dtype=tf.float32 ,  initializer= [2.])
不能这么写，因为，会导致 参数功能冲突，报错如下：
    raise ValueError("If initializer is a constant, do not specify shape.")
ValueError: If initializer is a constant, do not specify shape.

TensorFlow中tf.float32 和"float"的区别是什么
数位的区别，一个在内存中占分别32和64个bits，
也就是4bytes或8bytes 数位越高浮点数的精度越高
'''
var1 = tf.get_variable(name='var2', shape=[1], dtype=tf.float32)
var3 = tf.get_variable(name='var3', shape=[1], dtype=tf.float32)
var5 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
var6 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name, sess.run(var1))
    print(var3.name, sess.run(var1))
    print(var5.name, sess.run(var5))
    print(var6.name, sess.run(var6))
    '''
    var2:0 [-0.18272138]
    var3:0 [-0.18272138]
    var2_1:0 [2.]
    var2_2:0 [2.]
    '''

'''
get_variable 这样的写法，不能重名 ， 下边写法可以重名
tf.get_variable 一般和 tf.variable_scope 配合使用，用于在同一个的变量域中共享同一个变量，
'''
with tf.variable_scope('scope1') as scope1:
    a1 = tf.Variable(1, name='w1')
    print(a1.name)   # a1变量的名字：scope1/w1:0

    a2 = tf.get_variable(name='a3', initializer=1.0)
    print(a2.name)   # a2变量的名字：scope1/a3:0

    tf.get_variable_scope().reuse_variables() # 或用 scope1.reuse_variables()
    # Reuse 重复使用,重用,再利用

    a3 = tf.get_variable(name='a3', initializer=2.0)
    print(a3.name)   # a3变量的名字：scope1/a3:0

init = tf.global_variables_initializer()  #输出数据时必须要全局初始化

with tf.Session() as sess:
    sess.run(init)
    print (sess.run(a1)) # a1变量的值：1
    print (sess.run(a2)) # a2变量的值：1.0
    print (sess.run(a3)) # a3变量的值：1.0 a3使用a2的值

print('---------------------')
'''
或在不同变量域中使用同名变量。
'''
with tf.variable_scope("scope2") as scope2:
    a2 = tf.get_variable(name='a3', initializer=1.0)
with tf.variable_scope(scope2, reuse = True):
    a3 = tf.get_variable(name='a3', initializer=2.0)

print(a2.name) # scope2/a3:0
print(a3.name) # scope2/a3:0

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print (sess.run(a2)) # a2变量的值：1.0
    print (sess.run(a3)) # a3变量的值：1.0    a3使用scope2中的a2值

print('---------------------')
'''
tf.get_variable(name,  shape, initializer):
name就是变量的名称，shape是变量的维度，initializer是变量初始化的方式，初始化的方式有以下几种：

tf.constant_initializer：常量初始化函数
tf.random_normal_initializer：正态分布
tf.truncated_normal_initializer：截取的正态分布   # truncated 缩短,截取,截断
tf.random_uniform_initializer：均匀分布
tf.zeros_initializer：全部是0
tf.ones_initializer：全是1
tf.uniform_unit_scaling_initializer：满足均匀分布，但不影响输出数量级的随机值
'''
a1 = tf.get_variable(name='a1', shape=[2, 3], initializer=tf.random_normal_initializer(mean=0, stddev=1))
a2 = tf.get_variable(name='a2', shape=[1], initializer=tf.constant_initializer(1))
a3 = tf.get_variable(name='a3', shape=[2, 3], initializer=tf.ones_initializer())

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(a1))
    print(sess.run(a2))
    print(sess.run(a3))


'''
tf.get_variable() 不受tf.name_scope约束
'''
with tf.name_scope("my_scope"):
    v1 = tf.get_variable("a", [1], dtype=tf.float32)
    v2 = tf.Variable(1, name="b", dtype=tf.float32)
    mysum = tf.add(v1, v2)

print(v1.name)      # a:0
print(v2.name)      # my_scope/b:0
print(mysum.name)   # my_scope/Add:0

'''
tf.get_variable() 都受tf.variable_scope约束
'''
with tf.variable_scope("my_scope2"):
    v1 = tf.get_variable("a", [1], dtype=tf.float32)
    v2 = tf.Variable(1, name="b", dtype=tf.float32)
    mysum = tf.add(v1, v2)

print(v1.name)      # my_scope/a:0
print(v2.name)      # my_scope/b:0
print(mysum.name)   # my_scope/Add:0