# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf

"""
   tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。
tf.nn.embedding_lookup（tensor, id）:tensor就是输入张量，id就是张量对应的索引
"""
a = np.random.random([10, 4])
print(a)
print("-"*40)
b = tf.nn.embedding_lookup(a, [1, 3])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b))