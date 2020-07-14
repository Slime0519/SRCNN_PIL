import torch.nn as nn
import torch.nn.functional as F
lr = 1e-4

class SRCNN(nn.Module):
    def __init__(self,c=1,n1=64,n2=32,f1=9,f3=5):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=n1, kernel_size=f1, bias='True',padding=4)
        self.conv2 = nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=1, bias='True', padding=0)
        self.conv3 = nn.Conv2d(in_channels=n2, out_channels=c, kernel_size=f3, bias='True', padding=2)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)
    """
        self.Model = nn.Sequential()
        self.Model.add_module("patch_extraction",self.conv1)
        self.Model.add_module("relu1",nn.ReLU())
        self.Model.add_module("non-linear mapping",self.conv2)
        self.Model.add_module("relu2",nn.ReLU())
        self.Model.add_module("reconstruction",self.conv3)
    """
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        return out
