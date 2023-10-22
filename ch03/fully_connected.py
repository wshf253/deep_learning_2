import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul

'''
c = np.array([[1, 0, 0, 0, 0, 0, 0]])  # 2d array for mini batch
W = np.random.randn(7, 3)
h = np.matmul(c, W)
print(h)
'''
c = np.array([[1, 0, 0, 0, 0, 0, 0]])
W = np.random.randn(7, 3)
layer = MatMul(W)
h = layer.forward(c)
print(h)