import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt

from trainDataPD import getData

import config as conf


conf._init()
conf.set_value("issue_num", 70)
issue_num = conf.get_value("issue_num")


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")

    return X, Y


def initialize_parameters(n_x, n_y, layer1Num, layer2Num):

    tf.set_random_seed(1)  # 指定随机种子

    W1 = tf.get_variable("W1", [layer1Num, n_x], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [layer1Num, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [layer2Num, layer1Num], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [layer2Num, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [n_y, layer2Num], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [n_y, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    # Z1 = tf.matmul(W1,X) + b1             #也可以这样写
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3
    A3 = tf.nn.relu(Z3)
    return A3


def compute_cost(A3, Y):
    # y_ = tf.transpose(A3)  # 转置
    # y = tf.transpose(Y)  # 转置

    cost = tf.reduce_mean(tf.square(Y - A3))
    return cost


def model(X_train, Y_train, layer1Num, layer2Num, learning_rate=0.001, num_epochs=10000, print_cost=True, is_plot=True):

    ops.reset_default_graph()  # 能够重新运行模型而不覆盖tf变量
    tf.set_random_seed(1)

    (n_x, m) = X_train.shape  # 获取输入节点数量和样本数
    n_y = Y_train.shape[0]  # 获取输出节点数量

    costs = []  # 成本集

    # 给X和Y创建placeholder
    X, Y = create_placeholders(n_x, n_y)

    # 初始化参数
    parameters = initialize_parameters(n_x, n_y, layer1Num, layer2Num)

    # 前向传播
    A3 = forward_propagation(X, parameters)

    # 计算成本
    cost = compute_cost(A3, Y)

    # 反向传播，使用Adam优化
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # 初始化所有的变量
    init = tf.global_variables_initializer()

    # 开始会话并计算
    with tf.Session() as sess:
        # 初始化
        sess.run(init)

        # 正常训练的循环
        for epoch in range(num_epochs):

            epoch_cost = 0  # 每代的成本

            # 数据已经准备好了，开始运行session
            _, epoch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})

            if epoch_cost < 0.1:
                break

            # 记录并打印成本
            # # 记录成本
            if epoch % 5 == 0:
                costs.append(epoch_cost)
                # 是否打印：
                if print_cost and epoch % 100 == 0:
                    print("epoch = " + str(epoch) + "    epoch_cost = " + str(epoch_cost))

        # 是否绘制图谱
        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        # 保存学习后的参数
        parameters = sess.run(parameters)
        print("参数已经保存到session。")

        return parameters


x_train, y_train, x_test = getData(ball=1)
parameters = model(x_train, y_train, layer1Num=64, layer2Num=16, is_plot=False)
x = tf.placeholder(tf.float32, [1*issue_num, 1])
y_ = forward_propagation(x, parameters)
sess = tf.Session()
prediction = sess.run(y_, feed_dict={x: x_test})
print(prediction)
