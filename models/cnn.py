import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet161

class Identity(nn.Module):
    """
        Identity layer, does nothing
    """
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x

class ConvNet(nn.Module):
    """
        The Conv feature extractor based on densenet    
    """
    def __init__(self, out_dim, pretrained=True):
        super(ConvNet, self).__init__()
        self.densenet = densenet161(pretrained=pretrained)
        self.densenet.classifier = Identity()
        self.pooling = nn.AdaptiveAvgPool1d(out_dim)

    def forward(self, x):
        x = self.densenet(x)
        x.unsqueeze_(1)
        print(x.shape)
        x = self.pooling(x)
        return x