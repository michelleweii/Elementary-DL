# -*- coding: utf-8 -*-
# 定义了神经网络的训练过程
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

# 加载mnist_inference.py中定义的常量和前向传播的参数。
from Unit5_BestCase import mnist_inference

# 配置神经网络的参数。
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
# 模型保存的路径和文件名
MODEL_SAVE_PATH = 'Unit5_BestCase/path'
MODEL_NAME = 'best_model.ckpt'

def train(mnist):
    # 定义输入输出placeholder
    x = tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 直接使用mnist_inference.py中定义的前向传播过程。
    y = mnist_inference.inference(x.regularizer)
    global_step = tf.Variable(0,trainable=False)

    # 和5.2.1小节样例中类似地定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase='True'
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')

# 初始化Tensorflow持久化类
saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # 在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序来完成。
    for i in range(TRAINING_STEPS):
        xs, ys = mnist.train.next_batch(BATCH_SIZE)
        _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})

