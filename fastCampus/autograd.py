import torch
import torch.nn.functional as F
x = torch.FloatTensor([[1, 2],
                       [3, 4]]).requires_grad_(True)
x1 = x + 2
x2 = x - 2
x3 = x1 * x2
y = x3.sum()
print(y.backward())

target = torch.FloatTensor([[.1, .2, .3],
                            [.4, .5, .6],
                            [.7, .8, .9]])
