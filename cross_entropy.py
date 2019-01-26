import tensorflow as tf
# 2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 假设词汇表的大小为3，语料包含两个单词"2 0"
word_labels = tf.constant([2,0])

# 假设模型对两个单词预测时，产生的logit分别是[2.0，-1.0，3.0]和[1.0，0.0，-0.5]。这里的logit不是概率。
# 如果需要计算概率，则需要调用prob=tf.nn.softmax(logits)。但这里计算交叉熵的函数直接输入logits即可。
predict_logits = tf.constant([[2.0,-1.0,3.0],[1.0,0.0,-0.5]])

# 使用tf.nn.sparse_softmax_cross_entropy_with_logits计算交叉熵。
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=word_labels,logits=predict_logits)

# 运行程序，计算loss的结果是[ 0.32656264,  0.46436879]
with tf.Session() as sess:
    print(loss.eval())

#
word_prob_distribution = tf.constant([[0.0,0.0,1.0],[1.0,0.0,0.0]])
loss = tf.nn.softmax_cross_entropy_with_logits(labels=word_prob_distribution,logits=predict_logits)
# [ 0.32656264  0.46436879]
with tf.Session() as sess:
    print(sess.run(loss))
    