import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


#📍 (A) Simple Fully Connected Network (nn1)
#Baaseline model
#Fully connected layers with ReLU activation
#No convolutional layers
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



# 📍 (B) Convolutional Neural Network (CNNModel)
# Two convolutional layers with max pooling.
# Extracts spatial features from X-rays.
# Better than the fully connected network.
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


# 📍 (C) ResNet-18 (From Scratch & Pretrained)
# ResNet from Scratch
# Pretrained ResNet with Transfer Learning
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



# 1️⃣ Add Batch Normalization & Dropout
# Batch Normalization helps in stabilizing training, while Dropout prevents overfitting.
class CNNModelAdvanced(nn.Module):
    def __init__(self):
        super(CNNModelAdvanced, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch Normalization
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)  # Dropout Layer
        self.fc1 = nn.Linear(64 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(torch.relu(self.fc1(x)))  # Apply dropout
        x = self.fc2(x)
        return x

# 2️⃣ Try a Deeper ResNet (ResNet-50 or DenseNet)
# Instead of ResNet-18, let's try ResNet-50, which has more layers and can extract deeper features.
class ResNet50Pretrained(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50Pretrained, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Alternatively, we can try DenseNet, which is even more efficient in feature extraction:
# ✅ DenseNet keeps feature maps from previous layers → Better gradient flow.
class DenseNetPretrained(nn.Module):
    def __init__(self, num_classes=2):
        super(DenseNetPretrained, self).__init__()
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)

    def forward(self, x):
        return self.densenet(x)
