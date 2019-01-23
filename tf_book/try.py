import tensorflow as tf
import numpy as np
'''
sess = tf.InteractiveSession()
v = np.array([1, 2, 3, 4])
x = tf.constant(v, dtype=tf.float32, shape=[2, 2])
print(tf.reduce_mean(x).eval())
print(tf.reduce_mean(x, 0).eval())
print(tf.reduce_mean(x, 1).eval())
#END

v = tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]])
with tf.Session() as sess:
    print(tf.reduce_mean(v).eval())   #输出3.5

v = tf.constant([1.0, 2.0, 3.0])
with tf.Session() as sess:
    print(tf.log(v).eval())   #输出：[ 0.          0.69314718  1.09861231]

v1 = tf.constant([[1.0,2.0],[3.0,4.0]])
v2 = tf.constant([[5.0,6.0],[7.0,8.0]])
with tf.Session() as sess:
    print((v1*v2).eval())  #[[  5.  12.] [ 21.  32.]]
    print(tf.matmul(v1,v2).eval())  #[[ 19.  22.][ 43.  50.]]


import tensorflow as tf
v1 = tf.constant([1.0,2.0,3.0,4.0])
v2 = tf.constant([4.0,3.0,2.0,1.0])
sess = tf.InteractiveSession()
with tf.Session() as sess:
    #[False False  True  True]
    #[ 4.  3.  3.  4.]
    print(tf.greater(v1,v2).eval())
    print(tf.where(tf.greater(v1,v2),v1,v2).eval())
sess.close()
#end

import tensorflow as tf
from numpy.random import RandomState
batch_size = 8
# 两个输入节点
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
# 回归问题一般只有一个输出节点
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")
# 定义了一个单层的神经网络前向传播的过程，这里就是简单加权和
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)
# 定义预测多了和预测少了的成本
loss_less = 10
loss_more = 1
#在windows下，下面用这个where替代，因为调用tf.select会报错
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_)*loss_more, (y_-y)*loss_less))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
#通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
"""
设置回归的正确值为两个输入的和加上一个随机量，之所以要加上一个随机量是
为了加入不可预测的噪音，否则不同损失函数的意义就不大了，因为不同损失函数
都会在能完全预测正确的时候最低。一般来说，噪音为一个均值为0的小量，所以
这里的噪音设置为-0.05， 0.05的随机数。
"""
Y = [[x1 + x2 + rdm.rand()/10.0-0.05] for (x1, x2) in X]
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    steps = 5000
    for i in range(steps):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
    print(sess.run(w1))
#end


import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

#两个输入节点
x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
#回归问题一般只有一个输出节点
y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input')

#定义了一个单层的神经网络前向传播的过程，这里就是简单加权和
w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y = tf.matmul(x,w1)

#定义预测多了和预测少了的成本
loss_less = 10
loss_more =1
loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*loss_more,(y_-y)*loss_less))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
#设置回归的正确值为两个输入的和加上一个随机量。之所以要加上一个随机量是为了加入不可预测的噪音，否则不同损失函数的意义就不大了，
#因为不同损失函数都会在能完全预测正确的时候最低。一般来说噪音为一个均值为0的小量，所以这里的噪音设置为-0.05 ~ 0.05的随机数
#
Y = [[x1+x2+rdm.rand()/10.0-0.05] for (x1,x2) in X]

#训练神经网络
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size) % dataset_size
        end = min(start+batch_size,dataset_size)
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        print(sess.run(w1))
#[[ 1.01934695]
#[ 1.04280889]]
'''
