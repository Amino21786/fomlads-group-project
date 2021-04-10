import numpy as np 
cm= np.array([[95,  8,  0,  0], [ 9 , 82, 17,  0], [ 0, 12, 71,  9], [ 0,  0,  7, 90]])
print(cm[1][1])
for i in range(4):
    precision=0
    fp=0
    tp=cm[i][i]
    for j in range(4):
        if i !=j:
          fp+=cm[i][j]
    precision=tp/(tp+fp)
    print("Precision for Price Range ", i, " is ",precision)