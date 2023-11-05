from common.np import *  # import numpy as np (or import cupy as np)
from common.layers import *
from common.functions import sigmoid

class GRU:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        # Wx - D x 3H, Wh - H x 3H, b - 3H
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None
    
    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        H, H3 = Wh.shape
        # r - reset gate, z - update gate -> same as forget, input gate in LSTM
        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H:2*H], Wx[:, 2*H:]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H:2*H], Wh[:, 2*H:]
        bz, br, bh = b[:H], b[H:2*H], b[2*H, 3*H]

        z = sigmoid(np.dot(x, Wxz) + np.dot(h_prev, Whz) + bz)
        r = sigmoid(np.dot(x, Wxr) + np.dot(h_prev, Whr) + br)
        h_hat = np.tanh(np.dot(x, Wxh) + np.dot(r*h_prev, Whh) + bh)
        h_next = (1 - z) * h_prev + z * h_hat
        
        self.cache = (x, h_prev, z, r, h_hat)
        return h_next
    
    def backward(self, dh_next):
        Wx, Wh, b = self.params
        H, H3 = Wh.shape
        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H:2*H], Wx[:, 2*H:]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H:2*H], Wh[:, 2*H:]
        x, h_prev, z, r, h_hat = self.cache

        dh_prev = dh_next * (1 - z)
        dh_hat = dh_next * z
        
        # tanh part
        dt = dh_hat * (1 - h_hat**2) # (1 - y**2)
        dWxh = np.dot(x.T, dt)
        dx = np.dot(dt, Wxh.T)
        dWhh = np.dot((r*h_prev).T, dt)
        dhr = np.dot(dt, Whh.T) # r * h_prev part
        dbh = np.sum(dt, axis=0)
        dh_prev += r * dhr

        # z, update gate
        dz = dh_next * h_hat - dh_next * h_prev # sum of 2 gardients
        dt = dz * (1 - z) * z # (1-y) * y
        dWxz = np.dot(x.T, dt)
        dx += np.dot(dt, Wxz.T)
        dWhz = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, Whz.T)
        dbz = np.sum(dt, axis=0)

        # r, reset gate
        dr = h_prev * dhr
        dt = dr * (1 - r) * r
        dWxr = np.dot(x.T, dt)
        dx += np.dot(dt, Wxr.T)
        dWhr = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, Whr.T)
        dbr = np.dum(dt, axis=0)

        self.grads[0][...] = np.hstack(dWxz, dWxr, dWxh)
        self.grads[1][...] = np.hstack(dWhz, dWhr, dWhh)
        self.grads[2][...] = np.hstack(dbz, dbr, dbh)

        return dx, dh_prev


class TimeGRU:
    def __init__(self, Wx, Wh, b, stateful=True):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.dh = None, None
        self.stateful = stateful

    def forward(self, xs):
        N, T, D = xs.shape
        H, H3 = self.params[1].shape # Wh.shape
        
        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.empty((N, H), dtype='f')
        
        for t in range(T):
            layer = GRU(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        
        return hs
    
    def backward(self, dhs):
        N, T, H = dhs.shape
        D, H3 = self.params[0].shape # Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        grads = [0, 0, 0]

        dh = 0
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh) # backward returns dx, dh_prev
            dxs[:, t, :] = dx
            for i, layer in enumerate(layer.grads):
                grads[i] += layer.grad
        
        for i, layer in enumerate(grads):
            self.grads[i][...] = grads[i]
        
        self.dh = dh
        return dxs

    def set_state(self, h):
        self.h = h
    
    def reset_state(self):
        self.h = None
