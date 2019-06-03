import pandas as pd
import numpy as np
import config as conf

# 设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


def getData(ball):
    """
    生成训练集（样本、label），生成待预测数据
    :param ball: 哪个球
    :return:
    """
    df = pd.read_csv("powerballData/sampleData-1.csv").sort_values(by=['time'], ascending=True)
    data_num = df.shape[0]
    issue_num = conf.get_value("issue_num")
    i_max = data_num - issue_num

    x_train = np.zeros(shape=(i_max, 1*issue_num))
    y_train = np.zeros(shape=(i_max, 1))

    for i in range(i_max):
        x = df.iloc[i: i+issue_num, ball].values.flatten()  # ndarray  1*(6*issue_num)
        y = df.iloc[i+issue_num, ball]
        x_train[i] = x
        y_train[i] = y

    x_train = x_train.T
    y_train = y_train.T
    x_test = df.iloc[data_num - issue_num: data_num, ball].values.flatten().reshape(-1, 1)

    print("---------训练集输入---------", x_train.shape)
    print("---------训练集输出---------", y_train.shape)
    print("---------待预测集输入---------", x_test.shape)

    return x_train, y_train, x_test
