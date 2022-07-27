import numpy as np

#投点个数
sizes = [5, 10, 20, 30, 40, 50, 60, 70, 80, 100]
#记录不同投点个数下的均值
mean = [0.0] * 10
#记录不同投点个数下的方差
var = [0.0] * 10

#对不同投点个数的情况进行实验
for num in range(0, 10):

    size = sizes[num]
    #每种投点个数的实验重复100次
    result = [0.0] * 100
    
    #每种投点个数的实验重复100次
    for time in range(0, 100):
        #生成随机数据点的横坐标x
        locations = np.random.rand(size, 1)
        #将生成的随机点横坐标代入函数进行累加积分
        for one in range(0, size):
            #该操作会使得使得随机数生成数组点变成0-1内的均匀有序随机分布
            x = locations[one]/size + one*1.0/size
            #进行累加积分
            result[time] += pow(x, 3)

        result[time] /= size
        
    #计算均值和方差
    results = np.array(result)
    mean[num] = results.mean()
    var[num] = results.var()

#打印结果
for i in range(0,8):
    print("投点个数为：%d，均值为：%.6f，方差为：%.8f，相对误差为：%.6f%%" %(sizes[i],\
                                                  decimal.Decimal("%.6f" %float(mean[i])),\
                                                  decimal.Decimal("%.8f" % float(var[i])),\
                                                  100*abs(decimal.Decimal("%.2f" %float(0.25))-decimal.Decimal("%.6f" %float(mean[i])))/(decimal.Decimal("%.2f" %float(0.25)))))
