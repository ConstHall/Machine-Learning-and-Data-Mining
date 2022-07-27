import numpy as np
rel = [4.0, 1.0, 0.0, 3.0, 4.0, 1.0, 0.0, 1.0, 0.0, 2.0]
DCG_5 = rel[0]
for i in range (1,5):
    DCG_5 = DCG_5 + rel[i]/np.log2(i+2)
print("%.6f" %DCG_5)