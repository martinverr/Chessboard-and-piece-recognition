import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

import os, glob
import numpy as np
import sys
sys.path.insert(0, './')
from full.models import *

## if you change the Model maybe it will not have the 'avgpool' layer and the fv maybe will not have lenght 512
lenght_feature_vector = 2048
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_saves_path = './scratch-cnn/modelsaves2/'
model_name = '2-ResNet50'

def print_name_and_lable(img_name):
    txt_file_path = os.path.join(dir_path, img_name.replace('.png', '.txt'))
    if os.path.exists(txt_file_path):
        with open(txt_file_path, 'r') as txt_file:
            print(img_name, ' ---> ', txt_file.read())
    else:
        print(f'no corresponding .txt file found for {img_name}')


def get_vector(image_name, model):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = model.transform(img).unsqueeze(0)
    # 3. Create a vector of zeros that will hold our feature vector
    my_embedding = torch.zeros(lenght_feature_vector)
    #    The 'avgpool' layer has an output size of length_feature_vector
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(o.data.size(1)))
    # 5. Attach that function to our selected layer
    h = model.model.avgpool.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding

def get_vectors(images_name, model):
        # Lista per memorizzare i vettori delle feature
    feature_vectors = np.zeros((len(images_name), model.model.avgpool.output_size))
    
    # Per ogni nome di immagine
    for image_name in image_names:
        # Carica l'immagine
        img = Image.open(image_name)
        # Applica le trasformazioni
        t_img = model.transform(img).unsqueeze(0)
        # Esegue il modello sull'immagine trasformata
        with torch.no_grad():
            output = model(t_img)
        # Ottieni il vettore delle feature
        feature_vector = model.model.avgpool.output.flatten()
        # Aggiungi il vettore delle feature alla lista
        feature_vectors.append(feature_vector)
    
    return feature_vectors

def generate_feature_vector(dir_path,model, save_on_file=True):
    num_png_files = len(glob.glob(dir_path + '/*.png'))
    num_png_files_path = [file for file in os.listdir(dir_path) if file.endswith('.png')]
    list_fv = np.zeros(shape=(num_png_files,lenght_feature_vector))
    for i,img_path in enumerate(num_png_files_path):
        if img_path.lower().endswith(".png"):
            fv = get_vector(os.path.join(dir_path,img_path), model)
            list_fv[i] = fv
        else:
            continue

    save_fv = torch.from_numpy(list_fv)
    
    if save_on_file:
        ## save tensor
        torch.save(save_fv, 'retrieval/feature_vector_pieces.pth')
    return save_fv

model = torch.load(f'{model_saves_path}{model_name}.pth', map_location=device)
dir_path = 'output/training_pieces'

# generate feature vector in a dir and save if you need, if you would like to load data comment this line
if False:
    generate_feature_vector(dir_path, model,save_on_file=False)

# load feature vector
feature_vector_tensor = torch.load('retrieval/feature_vector_pieces.pt')
# normalize both the new fv and the old ones
known_feature_vectors_normalized = torch.nn.functional.normalize(feature_vector_tensor, p=2, dim=1)

# the img for the query
new_feature_vector_normalized = get_vector('output/training_pieces/0000_A7.png', model)
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
[print_name_and_lable(img_name=path) for x,path in enumerate(num_png_files_path) if x in top_10_indices]