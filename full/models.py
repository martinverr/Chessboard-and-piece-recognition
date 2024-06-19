import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models as torchmodels
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

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
    
class ResNet18(nn.Module):
    """ResNet18 model.
    """
    transform = transforms.Compose([
        transforms.Resize((80, 160)),
        transforms.ToTensor()
        ])
    shape_input = (80,160)
    
    
    def __init__(self):
        super().__init__()
        #self.model = models.resnet18(pretrained=True)
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        n = self.model.fc.in_features
        self.model.fc = nn.Linear(n, 2)
        #for param in self.model.parameters():
        #  param.requires_grad = False

    def forward(self, x):
        return self.model(x)

class MaskRCNN_board(nn.Module):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    def __init__(self):
        super().__init__()
        num_classes = 2
        self.model = torchmodels.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
        
    def forward(self, x):
        return self.model(x)