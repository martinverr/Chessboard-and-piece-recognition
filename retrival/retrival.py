import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

import os, glob
import numpy as np

## if you change the Model maybe it will not have the 'avgpool' layer and the fv maybe will not have lenght 512

class ResNet(nn.Module):
    """ResNet model.
    """

    def __init__(self):
        super().__init__()
        #self.model = models.resnet18(pretrained=True)
        self.model = models.resnet18(weights=models.ResNet18_Weights)
        n = self.model.fc.in_features
        self.model.fc = nn.Linear(n, 2)
        #for param in self.model.parameters():
        #  param.requires_grad = False
        self.layer = self.model.avgpool

    def forward(self, x):
        return self.model(x)
    
    def get_layer(self):
        return self.layer
    

trasformer = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])


def get_vector(image_name, model):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(trasformer(img=img)).unsqueeze(0)
    # 3. Create a vector of zeros that will hold our feature vector
    my_embedding = torch.zeros(512)
    #    The 'avgpool' layer has an output size of 512
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(o.data.size(1)))
    # 5. Attach that function to our selected layer
    h = model.get_layer().register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding


def generate_feature_vector(dir_path,model, save_on_file=True):
    num_png_files = len(glob.glob(dir_path + '/*.png'))
    num_png_files_path = [file for file in os.listdir(dir_path) if file.endswith('.png')]
    list_fv = np.zeros(shape=(num_png_files,512))
    for i,img_path in enumerate(num_png_files_path):
        if img_path.lower().endswith(".png"):
            fv = get_vector(os.path.join(dir_path,img_path), model= model)
            list_fv[i] = fv
        else:
            continue

    save_fv = torch.from_numpy(list_fv)
    
    if save_on_file:
        ## save tensor
        torch.save(save_fv, 'feature_vector_pieces.pt')
    return save_fv

model = ResNet()
dir_path = 'output/training_pieces'

# generate feature vector in a dir and save if you need, if you would like to load data comment this line
generate_feature_vector(dir_path, model, save_on_file=False)

# load feature vector
feature_vector_tensor = torch.load('feature_vector_pieces.pt')
# normalize both the new fv and the old ones
known_feature_vectors_normalized = torch.nn.functional.normalize(feature_vector_tensor, p=2, dim=1)

# the img for the query
new_feature_vector_normalized = get_vector('output/training_pieces/0000_A2.png', model)
new_feature_vector_normalized = torch.nn.functional.normalize(new_feature_vector_normalized, p=2, dim=0)

# Compute cosine similarity between new_feature_vector and every feature vector in known_feature_vectors
similarities = torch.nn.functional.cosine_similarity(known_feature_vectors_normalized, new_feature_vector_normalized)

# Sort the indices based on cosine similarity in descending order
sorted_indices = torch.argsort(similarities, descending=True)

# Get the indices of the top 10 most similar images
top_10_indices = sorted_indices[:10]
print("Indices of the top 10 most similar images:", top_10_indices)

print("Path of the 10 most similar:")
num_png_files_path = [file for file in os.listdir(dir_path) if file.endswith('.png')]
_ = [print(path) for x,path in enumerate(num_png_files_path) if x in top_10_indices]