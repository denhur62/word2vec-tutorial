import torch
import torch.nn.functional as F


def mse(x_hat, x):
    y = ((x-x_hat)**2).mean()
    return y


x = torch.FloatTensor([[1, 1], [2, 2]])
x_hat = torch.FloatTensor([[0, 0], [0, 0]])

print(mse(x, x_hat))

print(F.mse_loss(x_hat, x))
