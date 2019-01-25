import numpy as np
import tensorflow as tf
'''
# 简单循环神经网络前向传播过程。
X = [1, 2]  # 本例有两个时刻
state = [0.0, 0.0]
# 分开定义不同输入部分的权重以方便操作
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
w_cell_input = np.asarray([0.5, 0.6])
b_cell = np.asarray([0.1, -0.1])

# 定义用于输出的全连接层参数。
w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1

# 按照时间顺序执行循环神经网络的前向传播过程。
for i in range(len(X)):
    # 计算循环体中的全连接层神经网路。
    before_activation = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_cell
    state = np.tanh(before_activation)

    # 根据当前时刻状态计算最终输出。
    final_output = np.dot(state, w_output) + b_output
    # 输出每个时刻的信息。
    print('before activation:', before_activation)
    print('state:', state)
    print('output:', final_output)
    
# before activation: [ 0.6  0.5]
# state: [ 0.53704957  0.46211716]
# output: [ 1.56128388]
# before activation: [ 1.2923401   1.39225678]
# state: [ 0.85973818  0.88366641]
# output: [ 2.72707101]

# 使用LSTM结构的循环神经网络的前向传播过程。

# 定义一个LSTM结构。在tensorflow中通过一句简单的命令就可以实现一个完整的LSTM结构。
# LSTM中使用的变量也会在该函数中自动被声明。
import tensorflow as tf
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)
# 将LSTM中的状态初始化为全0数组。和其他神经网络类似，在优化循环神经网络时，每次也会使用一个batch的训练样本。
# 以下代码中，batch_size给出了一个batch的大小
state = lstm.zero_state(batch_size=batch_size,tf.float32)

# 定义损失函数
loss = 0.0
# 虽然理论上循环神经网络可以处理任意长度的序列，但是在训练时为了避免梯度消散的问题，会规定一个最大的序列长度。
# num_steps来表示这个长度。
for i in range(num_steps):
#     在第一个时刻声明LSTM结构中使用的变量，在之后的时刻都需要复用之前定义好的变量。
    if i>0: tf.get_variable_scope().reuse_variables()
    # 每一步处理时间序列中的一个时刻。将当前输入（current_input）和前一时刻状
    # 态（state) 传入定义的LSTM结构可以得到当前LSTM结构的输出lstm_output和更新后的状态state。
    lstm_output,state = lstm(current_input,state)
    # 将当前时刻LSTM结构的输出传入一个全连接层得到最后的输出。
    final_output = tf.contrib.layers.fully_connected(lstm_output)
    # 计算当时时刻输出的损失
    loss += calc_loss(final_output,expected_output)
# end


# 深层循环神经网络
# 定义一个基本的LSTM结构作为循环体的基础结构。深层循环神经网络也支持使用其他的循环体结构。
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
# 通过MultiRNNCell类实现深层循环神经网络中每一个时刻的前向传播过程。
# 其中，number_of_layers表示了有多少层。
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm]*number_of_layers)

# 和经典的循环神经网络一样，可以通过zero_state函数来获取初始状态。
state = stacked_lstm.zero_state(batch_size=batch_size,tf.float32)
# 计算每一时刻的前向传播结果。
for i in range(len(num_steps)):
    if i>0: tf.get_variable_scope().reuse_variables()
    stacked_lstm_output,state = stacked_lstm(current_input,state)
    final_output = tf.contrib.layers.fully_connected(stacked_lstm_output)
    loss += calc_loss(final_output,expected_output)

# 在tensorflow中实现带dropout的循环神经网络。
# 定义LSTM结构
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
# 使用DropoutWrapper类来实现dropout功能。该类通过两个参数来控制dropout的概率，一个参数为input_keep_prob,
# 用来控制输入的dropout的概率，output_keep_prob,用来控制输出的dropout概率。
dropout_lstm = tf.nn.rnn_cell.DropoutWrapper(lstm,output_keep_prob=0.5)
# 在使用了dropout的基础上定义
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([dropout_lstm]*number_of_layers)
'''
import codecs
import collections
from operator import itemgetter
# 存放原始数据的路径。
MODE = "PTB"    # 将MODE设置为"PTB", "TRANSLATE_EN", "TRANSLATE_ZH"之一。

if MODE == "PTB":             # PTB数据处理
    RAW_DATA = "../../datasets/PTB_data/ptb.train.txt"  # 训练集数据文件
    VOCAB_OUTPUT = "ptb.vocab"                         # 输出的词汇表文件
elif MODE == "TRANSLATE_ZH":  # 翻译语料的中文部分
    RAW_DATA = "../../datasets/TED_data/train.txt.zh"
    VOCAB_OUTPUT = "zh.vocab"
    VOCAB_SIZE = 4000
elif MODE == "TRANSLATE_EN":  # 翻译语料的英文部分
    RAW_DATA = "../../datasets/TED_data/train.txt.en"
    VOCAB_OUTPUT = "en.vocab"
    VOCAB_SIZE = 10000
