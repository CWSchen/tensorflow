import tensorflow as tf

'''
https://blog.csdn.net/huahuazhu/article/details/77161668

tf.nn.embedding_lookup(params,ids, partition_strategy=’mod’, name=None, validate_indices=True,max_norm=None)
根据ids中的id，寻找params中的对应元素,可以理解为索引，所以ids中元素值不能超出params的第一维的维数值。
比如，ids=[1,3,5]，则找出params中下标为1,3,5的向量组成一个矩阵返回。
参数说明：
params: 表示完整的embedding张量，或者除了第一维度之外具有相同形状的P个张量的列表，表示经分割的嵌入张量。
ids: 一个类型为int32或int64的Tensor，包含要在params中查找的id
'''
encode_embeddings = tf.constant([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]])  # 2*5
# input_ids中元素的值和encode_embeddings的第一维的维数有关，此例中为2维，input_ids只能是[0,2),也就是0和1
input_ids = tf.constant([[1, 1, 0], [1, 0, 1], [1, 0, 1], [0, 1, 1]])  # 4*3
session = tf.Session()
with session.as_default():
    # 结果results是4*3*5矩阵。
    results = tf.nn.embedding_lookup(encode_embeddings, input_ids)
    print(results.eval())  # tf.eval()函数用于显示张量tensor的值，但需要放在with session.as_default()中才能使用。