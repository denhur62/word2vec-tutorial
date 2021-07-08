import torch
import torch.nn as nn

w = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])

b = torch.FloatTensor([2, 2])

x = torch.FloatTensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])


# w,b가 학습이 안된다.
def linear(x, W, b):
    y = torch.matmul(x, W)+b
    return y


y = linear(x, w, b)
# print(y.size())

# 실제 구현 코드


class MyLinear(nn.Module):
    def __init__(self, input_dim=3, output_dim=2):
        self.input_dim = input_dim
        self.output_dim = output_dim

        super().__init__()
        self.w = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.b = nn.Parameter(torch.FloatTensor(output_dim))

    def forward(self, x):
        y = torch.matmul(x, self.w)+self.b

        return y


linear = MyLinear(3, 2)
# forward를 부르는 방법
y = linear(x)

print(y.size())
for p in linear.parameters():
    print(p)

linear = nn.Linear(3, 2)
y = linear(x)
print(y.size())
