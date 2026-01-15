import torch
import torch.nn as nn
import torchvision

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.cnn = torchvision.models.efficientnet_v2_m(pretrained=True).cuda()
