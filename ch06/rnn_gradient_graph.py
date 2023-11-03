import numpy as np
import matplotlib.pyplot as plt

N = 2 # batch_size
H = 3 # hidden_size
T = 20 # time_size

dh = np.ones((N, H))
np.random.seed(3)
Wh = np.random.randn(H, H) # 기울기 폭발
Wh = np.random.randn(H, H) * 0.5 # 기울기 소실

norm_list = []
for t in range(T):
    dh = np.matmul(dh, Wh.T)
    norm = np.sqrt(np.sum(dh**2)) / N
    norm_list.append(norm)

plt.plot(np.arange(len(norm_list)), norm_list)
plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
plt.xlabel('시간 크기(time step)')
plt.ylabel('노름(norm)')
plt.show()
