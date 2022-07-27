import numpy as np
import math

#积分函数公式
def calculate(x1, y1):
    cal_result = []
    for index in range(0, len(x)):
        temp1 = y1[index] ** 2 * math.exp(-(y1[index] ** 2))
        temp2 = x1[index] ** 4 * math.exp(-(x1[index] ** 2))
        temp3 = x1[index] * math.exp(-(x1[index] ** 2))
        cal_result.append((temp1 + temp2) / temp3)

    return cal_result

#投点个数
sizes = [10, 20, 30, 40, 50, 60, 70, 80, 100, 200, 500]
#记录不同投点个数下的均值
mean = []
#记录不同投点个数下的方差
var = []
floor_space = 2.0 * 2.0

for num in range(0, len(sizes)):
    size = sizes[num]
    result = []
    #每种投点个数的实验重复100次
    for time in range(0, 100):

        #生成随机数据点
        x = np.random.rand(size, 1)
        x *= 2
        x += 2

        y = np.random.rand(size, 1)
        y *= 2
        y -= 1
        temp = calculate(x, y)
        ones = np.array(temp)
        one = ones.mean() * floor_space
        result.append(one)

    results = np.array(result)
    mean.append(results.mean())
    var.append(results.var())
    #积分准确值
    true_value=112958.61998952225

#打印结果
for i in range(0,8):
    print("投点个数为：%d，均值为：%.6f，方差为：%.8f，相对误差为：%.4f%%" %(sizes[i],\
                                                  decimal.Decimal("%.6f" %float(mean[i])),\
                                                  decimal.Decimal("%.8f" % float(var[i])),\
                                                  100*abs(decimal.Decimal("%.6f" %float(true_value))-decimal.Decimal("%.6f" %float(mean[i])))/(decimal.Decimal("%.6f" %float(true_value)))))
