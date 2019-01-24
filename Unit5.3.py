import tensorflow as tf
'''
# 5.3 变量管理
# tf.get_variable函数来创建或者获取变量。当tf.get_variable用于创建变量时，它和tf.Variable的功能基本是等价的。
# 下面这两个定义是等价的
# v1 = tf.get_variable("v",shape=[1],initializer=tf.constant_initializer(1.0))
# v2 = tf.Variable(tf.constant(1.0,shape=[1],name='v'))

# 如果需要通过tf.get_variable获取一个已经创建的变量，需要通过tf.variable_scope函数来生
# 成一个上下文管理器，并明确指定在这个上下文管理器中，tf.get_variable将直接获取已经生成的变量。
# 通过tf.variable_scope函数来控制tf.get_variable函数获取已经创建过的变量。

# 在名字为foo的命名空间内创建名字为v的变量。
with tf.variable_scope('foo'):
    v = tf.get_variable('v',shape=[1],initializer=tf.constant_initializer(1.0))

## 因为在命名空间foo中已经存在名字为v的变量，所以下面的代码会报错：
#with tf.variable_scope('foo'):
#    v = tf.get_variable('v',[1])
## ValueError: Variable foo/v already exists, disallowed. Did you mean to set reuse=True in VarScope?

# 在生成上下文管理器时，将参数reuse设置为True。这样tf.get_variable函数将直接获取已经声明的变量。
with tf.variable_scope('foo',reuse=True):
    v1 = tf.get_variable('v',[1])
    print(v==v1)
# output:True，代表v,v1代表的是相同的tensorflow中变量。

# 将参数reuse设置为True时，tf.variable_scope将只能获取已经创建过得变量。
# 因为在命名空间bar中还没有创建变量v，所以下面的代码将会报错： ValueError: Variable bar/v does not exist,
# or was not created with tf.get_variable(). Did you mean to set reuse=None in VarScope?
# with tf.variable_scope('bar',reuse=True):
#    v = tf.get_variable('v',[1])
# end


# 当tf.variable_scope函数嵌套时，reuse参数的取值是如何确定的。
with tf.variable_scope('root'):
    # 可以通过tf.get_variable_scope().reuse函数来获取上下文管理器中reuse参数的取值。
    print(tf.get_variable_scope().reuse) # output:False,即最外层reuse是False.
    with tf.variable_scope('foo',reuse=True):
        print(tf.get_variable_scope().reuse)  # output:True
        with tf.variable_scope('bar'):
            print(tf.get_variable_scope().reuse)   # output:True
    print(tf.get_variable_scope().reuse)  # output:False
    
# end

# 如何通过tf.variable_scope来管理变量的名称。
v1 = tf.get_variable('v',[1])
print(v1.name)  # output: v:0
# 'v'为变量的名称，'：0'表示这个变量是生成变量这个运算的第一个结果。
with tf.variable_scope('foo'):
    v2 = tf.get_variable('v',[1])
    print(v2.name)  # output: foo/v:0
# 在tf.variable_scope中创建的变量，名称前面会加入命名空间的名称，并通过/来分割命名空间的名称和变量的名称。

with tf.variable_scope("foo"):
    with tf.variable_scope('bar'):
        v3 = tf.get_variable('v',[1])
        print(v3.name)  # foo/bar/v:0

    v4 = tf.get_variable('v1',[1])
    print(v4.name)  # foo/v1:0

# 创建一个名称为空的命名空间，并设置reuse=True。
with tf.variable_scope('',reuse=True):
    v5 = tf.get_variable('foo/bar/v',[1])
    print(v5 == v3)   # True
    v6 = tf.get_variable('foo/v1',[1])
    print(v6 == v4)   # True
    
#end
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784  # 输入节点
OUTPUT_NODE = 10  # 输出节点
LAYER1_NODE = 500  # 隐藏层数

BATCH_SIZE = 100  # 每次batch打包的样本个数

# 模型相关的参数
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 5000
MOVING_AVERAGE_DECAY = 0.99
'''
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):  # 5.2.1
    # 不使用滑动平均类
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2

    else:
        # 使用滑动平均类
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)
'''
'''
def inference(input_tensor, reuse=False):  # 5.3  没有添加滑动平均类
    # 定义第一层神经网络的变量和前向传播的结果。
    with tf.variable_scope('layer1',reuse=reuse):
        # 根据传进来的reuse判断是创建新变量还是使用已经创建好的。在第一次构造网络时需要创建
        # 新的变量，以后每次调用这个函数都直接使用reuse=False就不需要每次将变量传进来了。
        weights = tf.get_variable('weights',[INPUT_NODE,LAYER1_NODE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases',[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)

    # 类似地定义第二层神经网络的变量和前向传播过程。
    with tf.variable_scope('layer2',reuse=reuse):
        weights = tf.get_variable('weights',[LAYER1_NODE,OUTPUT_NODE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases',[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1,weights)+biases
    # 返回最后的前向传播结果。
    return layer2


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    # 生成隐藏层的参数。
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数。
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算不含滑动平均类的前向传播结果
    # y = inference(x, None, weights1, biases1, weights2, biases2) #5.2.1

    y = inference(x)  # 5.3
    # 定义训练轮数及相关的滑动平均类
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵及其平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 损失函数的计算
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    regularaztion = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularaztion

    # 设置指数衰减的学习率。
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 反向传播更新参数和更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 计算正确率
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话，并开始训练过程。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 循环的训练神经网络。
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc)))

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    train(mnist)

if __name__=='__main__':
    main()


# After 0 training step(s), validation accuracy using average model is 0.1366 
# After 1000 training step(s), validation accuracy using average model is 0.9778 
# After 2000 training step(s), validation accuracy using average model is 0.9828 
# After 3000 training step(s), validation accuracy using average model is 0.9838 
# After 4000 training step(s), validation accuracy using average model is 0.9846 
# After 5000 training step(s), test accuracy using average model is 0.9819


'''
'''
# 保存tensorflow计算图的方法。
v1 = tf.Variable(tf.constant(1.0,shape=[1]),name='v1')
v2 = tf.Variable(tf.constant(2.0,shape=[1]),name='v2')
result = v1 + v2
init_op = tf.global_variables_initializer()
# 声明tf.train.Saver类用于保存模型。
saver =  tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    #将模型保存到5.4.1model.ckpt文件。
    saver.save(sess,'path/5.4.1model.ckpt')



# 加载这个已经保存的tensorflow模型的方法。
# 使用和保存模型代码中一样的方式来声明变量。
v1 = tf.Variable(tf.constant(1.0,shape=[1]),name='v1')
v2 = tf.Variable(tf.constant(2.0,shape=[1]),name='v2')
result = v1+v2
saver = tf.train.Saver()
with tf.Session() as sess:
    #加载已经保存的模型，并通过已经保存的模型中变量的值来计算加法。
    saver.restore(sess,'path/5.4.1model.ckpt')
    print(sess.run(result))  # output: [ 3.]
# end

# 直接加载持久化的图。
saver = tf.train.import_meta_graph('path/5.4.1model.ckpt.meta')
with tf.Session() as sess:
    saver.restore(sess,'path/5.4.1model.ckpt')
    # 通过张量的名称来获取变量。
    print(sess.run(tf.get_default_graph().get_tensor_by_name('add:0'))) # output: [ 3.]


# 变量重命名是如何被使用的。
# 这里声明的变量名称和已经保存的模型中变量的名称不同。
v1 = tf.Variable(tf.constant(1.0,shape=[1]),name='other-v1')
v2 = tf.Variable(tf.constant(2.0,shape=[1]),name='other-v2')

# 如果直接使用tf.train.Saver()来加载模型会报变量找不到的错误。


v = tf.Variable(0,dtype=tf.float32,name='v')
# 在没有声明滑动平均模型时只有一个变量v,所以下面的语句只会输出"v:0".
for variables in tf.global_variables():
    print(variables.name)  # output: v:0
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())
# 在声明滑动平均模型之后，tensorflow会自动生成一个影子变量v/ExponentialMovingAverage。
# 于是下面的语句会输出"v:0"和"v/ExponentialMovingAverage:0"。
for variables in tf.global_variables():
    print(variables.name)

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    sess.run(tf.assign(v,10))
    sess.run(maintain_averages_op)
    saver.save(sess,'path/5.4.2.ckpt')
    print(sess.run([v,ema.average(v)])) # output: [10.0, 0.099999905]

v = tf.Variable(0,dtype=tf.float32,name='v')
# 通过变量重命名将原来变量v的滑动平均值直接赋给v。
saver = tf.train.Saver({'v/ExponentialMovingAverage':v})
with tf.Session() as sess:
    saver.restore(sess,'path/5.4.2.ckpt')
    print(sess.run(v)) # output: 0.0999999,这个值就是原来模型中变量v的滑动平均值。
# end


v = tf.Variable(0,dtype=tf.float32,name='v')
ema = tf.train.ExponentialMovingAverage(0.99)

# 通过使用variables_to_restore函数可以直接生成上面代码中提供的字典。
print(ema.variables_to_restore()) # {'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}

saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess,'path/5.4.2.ckpt')
    print(sess.run(v)) #0.0999999
    
# end

# tensorflow提供了convert_variables_to_constants函数，通过这个函数可以将计算图中的变量及其取值通过常量的方式保存。
from tensorflow.python.framework import graph_util

v1 = tf.Variable(tf.constant(1.0,shape=[1]),name='v1')
v2 = tf.Variable(tf.constant(2.0,shape=[2]),name='v2')
result = v1 + v2
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # 导出当前计算图的GraphDef部分，只需要这一部分就可以完成从输入层到输出层的计算过程。
    graph_def = tf.get_default_graph().as_graph_def()
    # 将图中的变量机器取值转化为常量，同时将图中不需要的节点去掉，
    # 最后一个参数['add']给出了需要保存的节点名称。add节点是上面定义的两个变量相加的操作。
    # 注意这里给出的是计算节点的名称，所以没有后面的：0.
    output_graph_def = graph_util.convert_variables_to_constants(sess,graph_def,['add'])
    # 将导出的模型存入文件。
    with tf.gfile.GFile('path/combined_model.pb','wb') as f:
        f.write(output_graph_def.SerializeToString())
    # output：Converted 2 variables to const ops.
#end

# 当只需要得到计算图中某个节点的取值时，这提供了一个更加方便的方法。
from tensorflow.python.platform import gfile
with tf.Session() as sess:
    model_filename = 'path/combined_model.pb'
    # 读取保存的模型文件，并将文件解析成对应的GraphDef Protocol Buffer.
    with gfile.FastGFile(model_filename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # 将graph_def中保存的图加载到当前的图中。return_elements=['add:0']给出了返回的张量的名称。
    # 在保存的时候给出的是计算节点的名称，所以为'add'。在加载的时候给出的是张量的名称，所以是add：0.
    
    result = tf.import_graph_def(graph_def,return_elements=['add:0'])
    print(sess.run(result))
    #output：[array([ 3.,  3.], dtype=float32)]
    
#end
'''

# 5.4.2持久化原理及数据格式
