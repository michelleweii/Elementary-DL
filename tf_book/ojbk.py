import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

A = np.array([[1, 2, 3], [4, 5, 6]])
x = tf.transpose(A, [1, 0])

B = np.array([[[1, 2, 3], [4, 5, 6]]])
y = tf.transpose(B, [2, 1, 0])
with tf.Session() as sess:
    print(A[1, 0])
    print(sess.run(x[0, 1]))
    print(B[0, 1, 2])
    print(sess.run(y[2, 1, 0]))

'''
4
4
6
6

'''

grads,_ = tf.clip