from common.np import *  # import numpy as np (or import cupy as np)
from common.layers import *
from common.functions import sigmoid
class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None
    
    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.matmul(h_prev, Wh) + np.matmul(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        x, h_prev, h_next = self.cache
        Wx, Wh, b = self.params

        dt = dh_next * (1 - h_next**2)
        dWh = np.matmul(h_prev.T, dt)
        dh_prev = np.matmul(dt, Wh.T)
        dWx = np.matmul(x.T, dt)
        dx = np.matmul(dt, Wx.T)
        db = np.sum(dt, axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev


class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.dh = None, None
        self.stateful = stateful

    def set_state(self, h):
        self.h = h
    
    def reset_state(self):
        self.h = None

    def forward(self, xs):
        Wx, Wh, b = self.params
        # N - batch_size, T - time, D - dimension of input, H - size of hidden layer
        N, T, D = xs.shape
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        
        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        
        return hs
    
    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0] # Wx, Wh, b
        for t in reversed(range(T)):
            layer = self.layers[T]
            dx, dh = layer.backward(dhs[:, t, :] + dh) # dhs from top, dh from next RNN (1 time future)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad
            
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs
    

class TimeEmbedding:
    # convert word id to word vector
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W
    
    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape # V - vocab_size

        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)
        
        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.bacward(dout[:, t, :])
            grad += layer.grads[0] 
            # Since grads = [dW], grad += layer.grads will cause error, layer.grads[0] will get dW from [dW]
        
        self.grads[0][...] = grad
        return None


class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
    
    def forward(self, x):
        N, T, D = x.shape # actually, x - (N, T, H) -> H is hidden size
        W, b = self.params # W - (D(H), V)

        rx = x.reshape(N*T, -1) # rx - (N*T, D)
        out = np.dot(rx, W) + b
        self.x = x
        return out.reshape(N, T, -1) # (N, T, V) -> V is vocab_size
    
    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N*T, -1) # dout - (N, T, V) -> (N*T, V)
        rx = x.reshape(N*T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout) # (D, N*T) dot (N*T, V) -> (D, V)
        dx = np.dot(dout, W.T) # (N*T, V) dot (V, D) -> (N*T, D)
        dx = dx.reshape(*x.shape) # (N, T, D)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V  = xs.shape

        if ts.ndim == 3:
            # when ts is one hot label - (N, T, V)
            ts = ts.argmax(axis=2) # ts - (N, T)

        mask = (ts != self.ignore_label) # if same as ignore label - 0, if not same - 1

        xs = xs.reshape(N*T, V)
        ts = ts.reshape(N*T)
        mask = mask.reshape(N*T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N*T), ts])
        ls *= mask # ignore_label에 해당하는 데이터는 손실을 0으로 설정 
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss
    
    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N*T), ts] -= 1 # y-t
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis] # ignore_label에 해당하는 데이터는 기울기를 0으로 설정

        dx = dx.reshape(N, T, V)

        return dx