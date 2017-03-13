from fastdtw import fastdtw
import numpy as np
import time
import random
import matplotlib.pyplot as plt

se0=np.random.random(size=2)
se1=np.random.random(size=2)
start0 = time.clock()
dist=np.sqrt((se0[0]-se1[0])**2+(se0[1]-se1[1])**2)
end0 = time.clock()
t0=end0-start0
t0=t0*1e7
print t0
normal=[]
for l in range(20):
    se_1=np.random.random(size=100*l)
    se_2=np.random.random(size=100*l)
    start = time.clock()

    fastdtw(se_1,se_2)

    end = time.clock()
    t1=(end-start)*1e5
    normal.append(t1)
plt.plot(normal)
plt.axhline(y=t0)
plt.show()