import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    # in_channels, out_channels, kernel_size, stride=1,
    # padding=0, dilation=1, groups=1, bias=True

    def __init__(self, num_class):
        super(Net, self).__init__()
        self.num_class = num_class
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 60)
        self.fc2 = nn.Linear(60, num_class)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x