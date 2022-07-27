import numpy as np
import math

#判断随机点是否在1/4圆范围内
def is_in(location):
    distance = location[0] ** 2 + location[1] ** 2
    distance = distance ** 0.5
    if distance < 1:
        return True
    else:
        return False

#投点个数
sizes = [20, 50, 100, 200, 300, 500, 1000, 5000]
#记录不同投点个数下的均值
mean = [0.0] * 8
#记录不同投点个数下的方差
var = [0.0] * 8

#对不同投点个数的情况进行实验
for num in range(0, 8):
    size = sizes[num]
    #每种投点个数的实验重复100次
    result = [0.0] * 100

    #每种投点个数的实验重复100次
    for time in range(0, 100):
        #生成随机坐标点size个(size代表投点个数)
        points = np.random.rand(size, 2)
        #判断每个点是否在1/4圆内
        for one in points:
            if is_in(one):
                result[time] += 1
        result[time] /= size
        result[time] *= 4

    #计算均值和方差
    results = np.array(result)
    mean[num] = results.mean()
    var[num] = results.var()
mean = np.array(mean)
var = np.array(var)

#打印结果
for i in range(0,8):
    print("投点个数为：%d，均值为：%.6f，方差为：%.8f，相对误差为：%.6f%%" %(sizes[i],\
                                                  decimal.Decimal("%.6f" %float(mean[i])),\
                                                  decimal.Decimal("%.8f" % float(var[i])),\
                                                  100*abs(decimal.Decimal("%.8f" %float(math.pi))-decimal.Decimal("%.6f" %float(mean[i])))/(decimal.Decimal("%.8f" %float(math.pi)))))

