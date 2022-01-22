# class TestClass:
#     def __init__(self):
#         print("init")

#     def __call__(self):
#         print("call")
#         return "called!"

# t = TestClass()

# # callによって関数のように扱える
# text = t()
# print(text)

class Variable:
    def __init__(self, data):
        self.data = data

# class Function:
#     def __call__(self, input):
#         x = input.data
#         y = x ** 2
#         output = Variable(y)
#         return output

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2

import numpy as np

x = Variable(np.array(10))
f = Square()
y = f(x)

print(type(y))
print(y.data)