import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
device = torch.device('cuda:0' if USE_CUDA else 'cpu')

boston = load_boston()

df = pd.DataFrame(boston.data, columns=boston.feature_names)
df["TARGET"] = boston.target

# 5개만
cols = ["TARGET", "INDUS", "RM", "LSTAT", "NOX", "DIS"]
# 부연 설명
# print(df[cols].describe())

data = torch.from_numpy(df[cols].values).float()
y = data[:, :1]
x = data[:, 1:]
n_epochs = 2000
learning_rate = 1e-3
print_interval = 100
model = nn.Linear(x.size(-1), y.size(-1))
optimizer = optim.SGD(model.parameters(),
                      lr=learning_rate)
for i in range(n_epochs):
    y_hat = model(x)
    loss = F.mse_loss(y_hat, y).to(device)

    optimizer.zero_grad()  # 그전에 있는 값들을 다 더한다.
    loss.backward()

    optimizer.step()  # 최적화 절차를 진행한다.

    if (i + 1) % print_interval == 0:
        print('Epoch %d: loss=%.4e' % (i + 1, loss))

df = pd.DataFrame(torch.cat([y, y_hat], dim=1).detach_().numpy(),
                  columns=["y", "y_hat"])

sns.pairplot(df, height=5)
plt.show()
