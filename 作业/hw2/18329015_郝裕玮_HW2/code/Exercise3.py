import matplotlib.pyplot as plt

#横纵坐标数据（Recall & Precision ）
precision = [1, 1, 2/3, 3/4, 4/5, 5/6, 5/7, 6/8, 6/9, 7/10]
recall = [1/7, 2/7, 2/7, 3/7, 4/7, 5/7, 5/7, 6/7, 6/7, 1]

#画图参数设置
plt.title("Precision-Recall(PR) Curve", fontsize=16)
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.tick_params(axis='both', labelsize=12)
plt.plot(recall, precision, linewidth=3)
plt.show()