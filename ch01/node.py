import numpy as np

# Repeat node - N개로의 분기(반복) 노드
D, N = 8, 7
x = np.random.randn(1, D)
y = np.repeat(x, N, axis=0) # forward

dy = np.random.randn(N, D)
dx = np.sum(dy, axis=0, keepdims=True) # backward
# if keepdims=True x shape will be (1, D), else x shape will be (D)

# Sum node - N x D 배열에 대해 그 합을 0축에 대해 구한다
D, N = 8, 7
x = np.random.randn(N, D)
y = np.sum(x, axis=0, keepdims=True) # forward - y.shape = (1, D)

dy = np.random.randn(1, D)
dx = np.repeat(y, N, axis=0) # backward
# sum - repeat node 반대 관계

# MatMul node
class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
    
    def forward(self, x):
        W, = self.params
        self.x = x
        out = np.matmul(self.x, W)
        return out
    
    def backward(self, dout):
        W, = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        self.grads[0][...] = dW
        # grads[0] = dW -> shallow copy, grads[0][...] -> deep copy
        return dx