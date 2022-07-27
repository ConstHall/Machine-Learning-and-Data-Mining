import matplotlib.pyplot as plt
import numpy as np
import math

# 计算两个向量间的距离
def dis(a, b, ax):
    #计算向量之间距离，axis = 1代表按行向量进行计算处理
    return np.linalg.norm(a-b, axis = ax)

# K_means聚类迭代算法
def K_means(X, center, total_times):
    '''
        X：需要进行分类的多个向量
        center：聚类的初始中心向量（红绿蓝3个聚类中心）
        total_times: 迭代次数
    '''
    # 储存更新后的聚类中心坐标
    center_new = np.zeros(center.shape)
    # 储存每个点所在的聚类(即红绿蓝3个聚类中心)
    X_cluster = np.zeros(len(X))

    #开始迭代
    times = 1
    while times < total_times :
        # 遍历X中的每个点
        for i in range(len(X)):
            # 计算当前点与3个中心点的距离
            distance_X = dis(X[i], center, 1)
            # 选出与该点最近的聚类中心点的下标
            cluster = np.argmin(distance_X)
            # 存储该点所在的聚类
            X_cluster[i] = cluster
        
        # 计算新的3个聚类中心点的坐标
        for i in range(3):
            # 用于存储每个聚类内部的点坐标
            cluster_point = []
            # 寻找X中相应聚类的点并将其统一存储到cluster_point中
            for j in range(len(X)):
                if X_cluster[j] == i:
                    cluster_point.append(X[j])
            cluster_point = np.array(cluster_point) 
            # 计算新的聚类中心点坐标
            center_new[i] = np.mean(cluster_point, axis=0)

        # 输出每次迭代的结果
        print("当前迭代次数为%d，各簇的中心点为：\n" %(times))
        center = center_new
        print("u1 = %s\nu2 = %s\nu3 = %s\n" %(center[0], center[1], center[2]))
        times = times + 1


X = np.array([[5.9, 3.2], [4.6, 2.9], [6.2, 2.8], [4.7, 3.2], [5.5, 4.2], [5.0, 3.0], [4.9, 3.1], [6.7, 3.1], [5.1, 3.8], [6.0, 3.0]])
center = np.array([[6.2, 3.2], [6.6, 3.7], [6.5, 3.0]])
K_means(X,center,5)
