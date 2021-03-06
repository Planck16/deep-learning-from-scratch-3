import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)

        output = Variable(y)
        output.set_creator(self)

        self.input = input
        self.output = output
        return output

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

A = Square()
B = Exp()
C = Square()

# 順伝播
x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 逆伝播
y.grad = np.array(1.0)

y.backward()
print(x.grad)

# C = y.creator
# b = C.input
# b.grad = C.backward(y.grad)

# B = b.creator
# a = B.input
# a.grad = B.backward(b.grad)

# A = a.creator
# x = A.input
# x.grad = A.backward(a.grad)
# print(x.grad)