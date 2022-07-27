import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import random
import math
from matplotlib.pyplot import MultipleLocator

# 逻辑回归公式
def logistic(w, x):
    # dot：矩阵乘法
    # 因为w和x都是a一维数组，所以返回的是向量的内积（一个数）
    temp = w.dot(x)
    temp = 1.0 / (1 + np.exp(-1 * temp))
    return temp

# 梯度下降
def gradient(train_data, label, rate, weight):
    sum = 0
    log_w = 0
    # 更新weight
    for i in range(len(train_data)):
        sum += (label[i] - logistic(weight, train_data[i])) * train_data[i]
    weight = weight + rate * sum

    return weight

# 利用训练后的weight对训练集和测试集计算错误率和最大似然函数值
def predict(data, label, weight):
    error = 0
    log_w = 0
    # 计算最大似然函数值
    for i in range(len(data)):
        temp = logistic(weight, data[i])
        log_w += label[i] * math.log(logistic(weight, data[i])) + (1-label[i]) * math.log(1 - logistic(weight, data[i]))
        # 预测错误则错误数+1
        if (temp >= 0.5 and label[i] == 0) or (temp < 0.5 and label[i] == 1):
            error = error + 1
    error /= len(data)
    return error, log_w


filename1 = "./dataForTrainingLogistic.txt"
filename2 = "./dataForTestingLogistic.txt"

# 训练集数据预处理
# train_num：训练集数据数量
train_num = len(open(filename1,'r').readlines())
read_file = open(filename1)
train_lines = read_file.readlines()

list_all_train = []
list_element_train  = []
list_label_train  = []
# 遍历每行数据
for train_line in train_lines:
    list1 = train_line.split()
    for one in list1:
        # 将字符串转为浮点数
        list_element_train.append(float(one))
    # list_all_train存储自变量（面积和距离，即X矩阵）
    list_all_train.append(list_element_train[0:6])
    # list_label_train存储因变量（房价，即结果矩阵）
    list_label_train.append(list_element_train[6])
    list_element_train = []
# 给list_all_train矩阵增加一个系数theta0，初始值设为1.0(因为对应着x0=1的系数，也就是偏差b)
for one in list_all_train:
    one.append(1.0)

# 均转为numpy数组类型，便于后续操作
train_data = np.array(list_all_train)
train_label = np.array(list_label_train)

# 测试集数据预处理
# test_num：训练集数据数量
test_num = len(open(filename2,'r').readlines())

read_test = open(filename2)
test_lines = read_test.readlines()

list_all_test = []
list_element_test = []
list_label_test = []
# 遍历每行数据
for test_line in test_lines:
    list2 = test_line.split()
    for one in list2:
        # 将字符串转为浮点数
        list_element_test.append(float(one))
    # list_all_test存储自变量（面积和距离，即X矩阵）
    list_all_test.append(list_element_test[0:6])
    # list_label_test存储因变量（房价，即结果矩阵）
    list_label_test.append(list_element_test[6])
    list_element_test = []
# 给list_all_test矩阵增加一个系数theta0，初始值设为1.0(因为对应着x0=1的系数，也就是偏差b)
for one in list_all_test:
    one.append(1.0)

# 均转为numpy数组类型，便于后续操作
test_data = np.array(list_all_test)
test_label = np.array(list_label_test)


''' 
    参数初始化：
    weight：权重因子
    learn_rate：学习率
    x：自变量矩阵
    train_loss：训练集误差
    test_loss：测试集误差
'''
weight = np.zeros(7)
# 学习率
learn_rate = 0.00015
x = []
# 最大化似然函数值
train_log_w = []
test_log_w = []
# 误差（这里指错误率）
train_error = []
test_error = []

# 开始迭代学习
for i in range(10,400,10):
    weight = np.zeros(7)
    sample = []
    sample_label = []
    print("\n\n当前训练集大小为：%d\n\n" % i)
    for j in range(0,i):
        m = random.randint(0, len(train_data) - 1)
        sample.append(train_data[m])
        sample_label.append(train_label[m])
        sample1 = np.array(sample)
        sample_label1 = np.array(sample_label)
    for k in range(1, 201):
        weight = gradient(sample1, sample_label1, 0.00015, weight)
        train_error1, train_log_w1 = predict(train_data, train_label, weight)
        test_error1, test_log_w1 = predict(test_data, test_label, weight)
        if (k % 20 == 0):
            print("当前迭代次数为：%d，训练集误差为：%f，测试集误差为：%f" % (k, train_error1, test_error1))
    train_error.append(train_error1)
    test_error.append(test_error1)
    x.append(i)


# 画图
plt.rcParams['font.sans-serif'] = ['SimHei'] # 替换sans-serif字体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题
plt.figure(figsize=(4,3),dpi=300)
plt.xlabel('训练集大小')
plt.ylabel('误差')
plt.plot(x, train_error,label='测试集')
plt.plot(x, test_error,label='训练集')
plt.legend()

plt.show()
