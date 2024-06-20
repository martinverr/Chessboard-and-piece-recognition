import shutil
import numpy as np
import random
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


# Definisci il numero desiderato di immagini per classe
num_images_per_class = 100
num_images_tot = 1200
lenght_feature_vector= 512


for filename in os.listdir('./output/training_pieces'):
    if filename.endswith('.png'):
        txt_filename = os.path.join('./output/training_pieces/', f"{os.path.splitext(filename)[0]}.txt")

        with open(txt_filename, 'r') as file:
            contenuto = file.read()
            if not os.path.exists(f'./output/training_pieces_divided_into_classes/{contenuto}'):
                os.makedirs(f'./output/training_pieces_divided_into_classes/{contenuto}')
            shutil.copy(os.path.join('./output/training_pieces/', filename), os.path.join(f'./output/training_pieces_divided_into_classes/{contenuto}', filename))

for dir in os.listdir('./output/training_pieces_divided_into_classes'):
        piece_dir = os.path.join('./output/training_pieces_divided_into_classes/', dir)
        
        # Controlla se l'elemento è una cartella
        if os.path.isdir(piece_dir):
            # Utilizza len(os.listdir()) per contare il numero di elementi nella cartella
            print(f'{dir}: {len(os.listdir(piece_dir))}')



# Crea un array numpy vuoto
image_paths_array = np.empty((0,))

# Itera attraverso le cartelle delle classi
for i, dir in enumerate(os.listdir('./output/training_pieces_divided_into_classes')):
    class_dir = os.path.join('./output/training_pieces_divided_into_classes/', dir)
    
    # Controlla se l'elemento è una cartella
    if os.path.isdir(class_dir):
        image_files = os.listdir(class_dir)
        
        # Seleziona casualmente num_images_per_class immagini dalla cartella corrente
        selected_images = random.sample(image_files, min(num_images_per_class, len(image_files)))
        
        # Aggiungi i percorsi delle immagini all'array numpy
        image_paths_array = np.append(image_paths_array, [os.path.join(class_dir, image) for image in selected_images])


## if you change the Model maybe it will not have the 'avgpool' layer and the fv maybe will not have lenght 512
lenght_feature_vector = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_saves_path = './scratch-cnn/modelsaves2/'
model_name = 'ResNet18_80x160'

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


def generate_feature_vector_file(image_paths_array, model, save_on_file=False):

    list_fv = list_fv = np.zeros(shape=(num_images_tot, lenght_feature_vector))
    classes = {
    'b_Pawn': 0, 'b_Bishop': 1, 'b_Knight': 2, 'b_Rook': 3, 
    'b_Queen': 4, 'b_King': 5, 'w_King': 6, 'w_Queen': 7, 
    'w_Rook': 8, 'w_Knight': 9, 'w_Bishop': 10, 'w_Pawn': 11
}

    for i, image_path in enumerate(image_paths_array):
        if image_path is not None:
            if os.path.exists(image_path) == True:
                fv = get_vector(image_path, model)
                list_fv[i] = fv
            else:
                raise Exception('image_path non esistente')

    classes_list = [os.path.basename(os.path.dirname(path)) for path in image_paths_array]
    nuova_lista = [classes[key] for key in classes_list]

    data = np.column_stack((nuova_lista,list_fv)).astype(np.float32)
    data = torch.from_numpy(data)

    if save_on_file:
        ## save
        torch.save(data, './retrieval/feature_vector_pieces_resnet18.pt')
        return data



model = torch.load(f'{model_saves_path}{model_name}.pth', map_location=device)
generate_feature_vector_file(image_paths_array=image_paths_array, model=model, save_on_file=True)

print('File generato, il file contiene 1200 righe da 513 (in prima posizione c\'è il path dell\'immagine poi feature vector)')
