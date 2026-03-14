import numpy as np

A = np.zeros((25))
count=0
for i in range(1,6):
    for j in range(1,6):
        A[count] = i*j
        count+=1
A = A.reshape((5,5))
print(A)
B = np.zeros(75)
count = 0 
for i in range(1,6):
    for j in range(1,4):
        for k in range(1,6):
            B[count] = i * j * k
            count += 1
B = B.reshape((5,3,5))

pos = np.array([0,1,2]) 
shift = pos[np.newaxis, :, np.newaxis] * A[:, np.newaxis, :]
print(shift)