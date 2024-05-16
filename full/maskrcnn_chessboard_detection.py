import torch
from PIL import Image
import torchvision.transforms as T
from models import *
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


model = MaskRCNN_board()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('./maskRCNN_epoch_2_.pth', map_location = device))
model.to(device)
model.eval()

img = Image.open('./input/0024.png')
img = model.transform(img).unsqueeze(0)

with torch.no_grad():
    img = img.to(device)
    pred = model(img)

    
plt.imshow((pred[0]["masks"][0].cpu().detach().numpy() * 255).astype("uint8").squeeze())
plt.show()