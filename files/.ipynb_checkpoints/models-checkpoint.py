import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class nn1(nn.Module):
    def __init__(self):
        super(nn1, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(224 * 224, 128)  # Flattened input to 128 neurons
        self.fc2 = nn.Linear(128, 2)  # Binary classification output (Pneumonia vs Normal)
    
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class ResNetScratch(nn.Module):
    """ResNet-18 from Scratch (Modified for 1-Channel Input)"""
    def __init__(self, num_classes=2):
        super(ResNetScratch, self).__init__()
        self.resnet = models.resnet18(weights=None)  # No pretrained weights
        
        # Modify first convolution layer to accept 1-channel (grayscale) images
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # Modify output layer

    def forward(self, x):
        return self.resnet(x)


class ResNetPretrained(nn.Module):
    """ResNet-18 Pretrained (Transfer Learning, Modified for 1-Channel Input)"""
    def __init__(self, num_classes=2):
        super(ResNetPretrained, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Load pretrained model
        
        # Modify first convolution layer to accept 1-channel (grayscale) images
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # Modify output layer

    def forward(self, x):
        return self.resnet(x)