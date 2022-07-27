import numpy as np
import matplotlib.pyplot as plt

# 损失函数：均方误差
def loss(train_data, weight, train_real, num):
    # 将数据和权重两个矩阵相乘得到预测结果矩阵
    # matmul：将两个矩阵进行矩阵乘法
    result = np.matmul(train_data, weight.T)
    # 计算均方误差
    loss = train_real - result
    losses = pow(loss, 2)
    losses = losses.T
    final_loss = losses.sum()/(2*num)
    return final_loss

# 梯度下降
def gradient(train_data, weight, train_real, num):
    result = np.matmul(train_data, weight.T)
    loss = result - train_real
    x1 = train_data.T[0]
    x1 = x1.T
    theta_1 = loss * x1

    x2 = train_data.T[1]
    x2 = x2.T
    theta_2 = loss * x2

    x3 = train_data.T[2]
    x3 = x3.T
    theta_3 = loss * x3

    return theta_1.T.sum() / num, theta_2.T.sum() / num, theta_3.T.sum() / num

# 读取训练集和测试集数据
filename1 = "./dataForTrainingLinear.txt"
filename2 = "./dataForTestingLinear.txt"

# 训练集数据预处理
# train_num：训练集数据数量
train_num = len(open(filename1,'r').readlines())
read_file = open(filename1)
train_lines = read_file.readlines()

list_all_train = []
list_element_train  = []
list_real_train  = []
# 遍历每行数据
for train_line in train_lines:
    list1 = train_line.split()
    for one in list1:
        # 将字符串转为浮点数
        list_element_train.append(float(one))
    # list_all_train存储自变量（面积和距离，即X矩阵）
    list_all_train.append(list_element_train[0:2])
    # list_real_train存储因变量（房价，即结果矩阵）
    list_real_train.append(list_element_train[2])
    list_element_train = []
# 给list_all_train矩阵增加一个系数theta0，初始值设为1.0(因为对应着x0=1的系数，也就是偏差b)
for one in list_all_train:
    one.append(1.0)

# 均转为numpy数组类型，便于后续操作
train_data = np.array(list_all_train)
train_real = np.array(list_real_train)
# 矩阵转置
train_real = train_real.T

# 测试集数据预处理
# test_num：训练集数据数量
test_num = len(open(filename2,'r').readlines())

read_test = open(filename2)
test_lines = read_test.readlines()

list_all_test = []
list_element_test = []
list_real_test = []
# 遍历每行数据
for test_line in test_lines:
    list2 = test_line.split()
    for one in list2:
        # 将字符串转为浮点数
        list_element_test.append(float(one))
    # list_all_test存储自变量（面积和距离，即X矩阵）
    list_all_test.append(list_element_test[0:2])
    # list_real_test存储因变量（房价，即结果矩阵）
    list_real_test.append(list_element_test[2])
    list_element_test = []
# 给list_all_test矩阵增加一个系数theta0，初始值设为1.0(因为对应着x0=1的系数，也就是偏差b)
for one in list_all_test:
    one.append(1.0)

# 均转为numpy数组类型，便于后续操作
test_data = np.array(list_all_test)
test_real = np.array(list_real_test)
# 矩阵转置
test_real = test_real.T


''' 
    参数初始化：
    weight：权重因子
    learn_rate：学习率
    x：自变量矩阵（面积和距离）
    train_loss：训练集误差
    test_loss：测试集误差
'''
weight = np.array([0.0, 0.0, 0.0])
learn_rate = 0.00005
x = []
train_loss = []
test_loss = []

# 迭代1500000次
for num in range(1, 1500001):
    train_losses = loss(train_data, weight, train_real, train_num)
    test_losses = loss(test_data, weight, test_real, test_num)

    theta_1, theta_2, theta_3 = gradient(train_data, weight, train_real, train_num)
    theta = np.array([theta_1, theta_2, theta_3])
    # weight更新公式
    weight = weight - (theta * learn_rate)

    # 只打印每100000次的相关数据
    if num % 100000 == 0:
        x.append(num)
        train_loss.append(train_losses)
        test_loss.append(test_losses)
        print ("当前迭代次数为%d次，训练集误差为：%f，测试集误差为：%f"%(num, train_losses, test_losses))
        
# 画图
plt.rcParams['font.sans-serif'] = ['SimHei'] # 替换sans-serif字体
plt.figure(figsize=(4,3),dpi=300)
plt.xlabel('迭代次数')
plt.ylabel('误差')
plt.plot(x, train_loss,label='训练集')
plt.plot(x, test_loss,label='测试集')
plt.legend()
plt.show()
