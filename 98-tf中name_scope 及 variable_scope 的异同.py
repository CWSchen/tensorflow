import tensorflow as tf
'''
大意是说 name_scope及variable_scope的作用都是为了"不传引用而访问跨代码区域变量"的一种方式，
其内部功能是在其代码块内显式创建的变量都会带上scope前缀（如上面例子中的a），这一点它们几乎一样。
而差别是在其作用域中获取变量，它们对 tf.get_variable() 函数的作用是一个会自动添加前缀，一个不会添加前缀。
'''

with tf.name_scope("my_scope"):
    v1 = tf.get_variable("var1", [1], dtype=tf.float32)
    v2 = tf.Variable(1, name="var2", dtype=tf.float32)
    a = tf.add(v1, v2)

print(v1.name)  # var1:0
print(v2.name)  # my_scope/var2:0
print(a.name)   # my_scope/Add:0

print('-'*100)

with tf.variable_scope("my_scope"):
    v1 = tf.get_variable("var1", [1], dtype=tf.float32)
    v2 = tf.Variable(1, name="var2", dtype=tf.float32)
    a = tf.add(v1, v2)

print(v1.name)  # my_scope/var1:0
print(v2.name)  # my_scope/var2:0
print(a.name)   # my_scope/Add:0