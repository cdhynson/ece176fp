import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# Simple Fully Connected Network (baseline model)
# Fully connected layers with ReLU activation
# No convolutional layers
class nn1(nn.Module):
    def __init__(self):
        super(nn1, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(50176, 128)
        self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# Convolutional Neural Network (CNNModel)
# Two convolutional layers with max pooling
# Extracts spatial features from X-rays
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
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ResNet-18 (From Scratch & Pretrained)
# ResNet from Scratch
class ResNetScratch(nn.Module):
    """ResNet-18 from Scratch (Modified for 1-Channel Input)"""
    def __init__(self, num_classes=2):
        super(ResNetScratch, self).__init__()
        self.resnet = models.resnet18(weights=None)
        
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Pretrained ResNet with TL
class ResNetPretrained(nn.Module):
    """ResNet-18 Pretrained (Transfer Learning, Modified for 1-Channel Input)"""
    def __init__(self, num_classes=2):
        super(ResNetPretrained, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) 
        
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)



# Add Batch Normalization + Dropout
# Batch norm to help stabilize in training, while dropout will help prevent overfitting
class CNNModelAdvanced(nn.Module):
    def __init__(self):
        super(CNNModelAdvanced, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) 
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5) 
        self.fc1 = nn.Linear(64 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  
        x = self.dropout(torch.relu(self.fc1(x))) 
        x = self.fc2(x)
        return x


# ResNet-50, has more layers and can extract deeper features
class ResNet50Pretrained(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50Pretrained, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)