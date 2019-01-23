import tensorflow as tf
'''
#激活函数
#a = tf.nn.relu(tf.matmul(x,w1) + biases1)
#b = tf.nn.relu(tf.matmul(a,w2) + biases2)

#交叉熵
#cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))

v = tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]])
with tf.Session() as sess:
    print(tf.clip_by_value(v,2.5,4.5).eval())


v = tf.constant([1.0, 2.0, 3.0])
with tf.Session() as sess:
    print(tf.log(v).eval())   #输出：[ 0.          0.69314718  1.09861231]

v1 = tf.constant([[1.0,2.0],[3.0,4.0]])
v2 = tf.constant([[5.0,6.0],[7.0,8.0]])
with tf.Session() as sess:
    print((v1*v2).eval())  #[[  5.  12.] [ 21.  32.]]
    print(tf.matmul(v1,v2).eval())  #[[ 19.  22.][ 43.  50.]]

v = tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]])
with tf.Session() as sess:
    print(tf.reduce_mean(v).eval())   #输出3.5

#使用了softmax回归之后的交叉熵函数：
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y,y_)

#均方误差MSE
mse = tf.reduce_mean(tf.square(y_-y))

#自定义损失函数
loss = tf.reduce_sum(tf.select(tf.greater(v1,v2),(v1-v2)*a,(v1-v2)*b))


#tf.select函数(已被弃用，用tf.where代替)和tf.greater函数的用法
v1 = tf.constant([1.0,2.0,3.0,4.0])
v2 = tf.constant([4.0,3.0,2.0,1.0])
sess = tf.InteractiveSession()
with tf.Session() as sess:
    #[False False  True  True]
    #[ 4.  3.  3.  4.]
    print(tf.greater(v1,v2).eval())
    print(tf.where(tf.greater(v1,v2),v1,v2).eval())
sess.close()  #end

#通过一个简单的神经网络程序来讲解损失函数对模型训练结果的影响
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
#end

#在tensorflow中使用tf.train.exponential_decay（指数衰减学习率） P85
#decayed_learning_rate为每一轮优化时使用的学习率，
## learning_rate 为事先设定的初始学习率
#  decay_rate 为衰减系数
#  decay_steps 为衰减速度
# decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
global_step = tf.Variable(0)
#通过exponential_decay函数生成学习率
#learning_rate：0.1；staircase=True;则每100轮训练后要乘以0.96.
learning_rate = tf.train.exponential_decay(0.1,global_step,100,0.96,staircase=True)
#end


#正则化
weights =tf.constant([[1.0,-2.0],[-3.0,4.0]])
with tf.Session() as sess:
    #L1正则化
    print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))  #output:5.0
    #L2正则化
    print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))   #output:7.5

#end

#通过集合计算一个5层神经网络的带L2正则化的损失函数

#当网络结构复杂之后定义网络结构的部分和计算损失函数的部分可能不在同一个函数中
#获取一层神经网络边上的权重，并将这个权重的L2正则化损失加入名称为'losses'
def get_weight(shape,lambda1):
    #生成一个变量
    var = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    #add_to_collection函数将这个新生成变量的L2正则化损失项加入集合。
    #这个函数的第一个参数'losses'是集合的名字，第二个参数是要加入这个集合的内容。
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lambda1)(var))
    #返回生成的变量
    return var

x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))
batch_size = 8
#定义了每一层网络中节点的个数
layer_dimension = [2,10,10,10,1]
#神经网络的层数
n_layers= len(layer_dimension)

#这个变量维护前向传播时最深层的节点，开始的时候就是输入层
cur_layer = x
#当前层的节点个数
in_dimension = layer_dimension[0]

#通过一个循坏来生成5层全连接的神经网络结构
for i in range(1,n_layers):
    #layer_dimension[i]为下一层的节点个数
    out_dimension = layer_dimension[i]
    #生成当前层中权重的变量，并将这个变量的L2正则化损失加入计算图上的集合
    weight = get_weight([in_dimension,out_dimension],0.001)
    bias = tf.Variable(tf.constant(0.1,shape=[out_dimension]))
    #使用RELU激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer,weight)+bias)
    #进入下一层之前将下一层的节点个数更新为当前层节点个数
    in_dimension = layer_dimension[i]

#在定义神经网络前向传播的同时已经将所有的L2正则化损失加入了图上的集合，
#在这里只需要计算刻画模型在训练数据上表现的损失函数。
mse_loss = tf.reduce_mean(tf.square(y_-cur_layer))

#将均方误差损失函数加入损失集合
tf.add_to_collection('losses',mse_loss)

#get_collection返回一个列表，这个列表是所有这个集合中的元素。在这个样例中，
#这些元素就是损失函数的不同部分，将它们加起来就可以得到最终的损失函数。
loss = tf.add_n(tf.get_collection('losses'))
#end
'''

#滑动平均模型

#定义一个变量用于计算滑动平均，这个变量的初始值为0。注意这里手动指定了变量的
#类型为tf.float32，因为所有需要计算滑动平均的变量必须是实数型。
v1 = tf.Variable(0,dtype=tf.float32)
#这里step变量模拟神经网络中迭代的论述，可以用于动态控制衰减率。
step = tf.Variable(0,trainable=False)

#定义一个滑动平均的类（class）。初始化时给定了衰减率（0.99）和控制衰减率的变量step。
ema = tf.train.ExponentialMovingAverage(0.99,step)
#定义一个更新变量滑动平均的操作。这里需要给定一个列表，每次执行这个操作时
#这个列表中的变量都会被更新。
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    #初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    #通过ema.average（v1）获取滑动平均之后变量的取值。在初始化之后变量V1的值和v1的滑动平均都为0.
    print(sess.run([v1,ema.average(v1)]))  #output：[0.0, 0.0]

    #更新变量v1的值到5.
    sess.run(tf.assign(v1,5))
    #更新v1的滑动平均值。衰减率为min{0.99,(1+step)/(10+step)=0.1}=0.1
    #所以v1的滑动平均会被更新为0.1*0+0.9*5=4.5
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)]))  #output: [5.0, 4.5]

    #更新step的值为10000.
    sess.run(tf.assign(step,10000))
    #更新v1的值为10.
    sess.run(tf.assign(v1,10))
    #更新v1的华东平均值。衰减率为min{0.99,(1+step)/(10+step)=0.999}}=0.99
    #所以v1的滑动平均会被更新为0.99*4.5+0.01*10=4.555
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)])) #output:[10.0, 4.5549998]

    #再次更新滑动平均值，得到的新滑动平均值为0.99*4.555+0.01*10=4.60945
    sess.run(maintain_averages_op)
    print(sess.run([v1,ema.average(v1)])) #output:[10.0, 4.6094499]

    #end

