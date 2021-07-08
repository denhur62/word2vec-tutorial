import torch

x = torch.FloatTensor(10, 4)

chunks = x.chunk(3, dim=0)

# for c in chunks:
#     print(c.size())


x = torch.randperm(3**3).reshape(3, 3, -1)
z1 = x.argmax(dim=0)
z2 = x.argmax(dim=1)
z3 = x.argmax(dim=2)
# print(x)
#print(x[0, :, 0])
# print(z1)
# print(z2)
# print(z3)
# values, indices = torch.sort(x, dim=1, descending=True)
# print(values)


# convert to CUDA
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:', device)
