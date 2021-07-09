import torch
import torch.nn.functional as F
x = torch.FloatTensor([[1, 2],
                       [3, 4]]).requires_grad_(True)
x1 = x + 2
x2 = x - 2
x3 = x1 * x2
y = x3.sum()
# print(y.backward())

target = torch.FloatTensor([[.1, .2, .3],
                            [.4, .5, .6],
                            [.7, .8, .9]])
x = torch.rand_like(target)
x.requires_grad = True
# print(x)

loss = F.mse_loss(x, target)
threshold = 1e-5
learning_rate = 1.
iter_cnt = 0

while loss > threshold:
    iter_cnt += 1

    loss.backward()  # Calculate gradients.

    x = x - learning_rate * x.grad

    # You don't need to aware this now.
    x.detach_()
    x.requires_grad_(True)

    loss = F.mse_loss(x, target)

    print('%d-th Loss: %.4e' % (iter_cnt, loss))
    print(x)
