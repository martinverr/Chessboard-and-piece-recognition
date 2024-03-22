import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models as torchmodels


class CNN_80x80_2Conv_2Pool_2FC(nn.Module):
    transform = transforms.Compose([
        transforms.Resize((80, 80)),
        transforms.ToTensor()
        ])
    shape_input = (80,80)
    
    def __init__(self):
        super(CNN_80x80_2Conv_2Pool_2FC, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 20 * 20, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)  # Two output classes: empty and no_empty

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        probabilities = F.softmax(x, dim=1)
        return probabilities
    


class ResNet50(nn.Module):
    """ResNet model.
    """
    transform = transforms.Compose([
        transforms.Resize((80, 160)),
        transforms.ToTensor()
        ])
    shape_input = (80,160)
    
    
    transform2 = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    shape_input2 = (224,224)
    
    def __init__(self):
        super().__init__()
        self.model = torchmodels.resnet50(weights=torchmodels.ResNet50_Weights.DEFAULT)
        n = self.model.fc.in_features
        self.model.fc = nn.Linear(n, 12)

    def forward(self, x):
        return self.model(x)
    
