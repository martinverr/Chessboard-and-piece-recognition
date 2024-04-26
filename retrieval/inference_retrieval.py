import torch
import os
import sys
import numpy as np
from PIL import Image
sys.path.insert(0, './')
from full.models import *

## if you change the Model maybe it will not have the 'avgpool' layer and the fv maybe will not have lenght 512
lenght_feature_vector = 2048
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_saves_path = './scratch-cnn/modelsaves2/'
model_name = '2-ResNet50'

classes = {'b_Bishop' : 0, 'b_King' : 1, 'b_Knight' : 2, 'b_Pawn' : 3, 'b_Queen' : 4,
                     'b_Rook' : 5, 'w_Bishop' : 6, 'w_King' : 7, 'w_Knight' : 8, 'w_Pawn' : 9, 'w_Queen' : 10,
                     'w_Rook' : 11}

inverted_classes = {value: key for key, value in classes.items()}

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_saves_path = './scratch-cnn/modelsaves2/'
model_name = '2-ResNet50'
model = torch.load(f'{model_saves_path}{model_name}.pth', map_location=device)

# load feature vector
loaded = torch.load('retrieval/feature_vector_pieces.pt')
feature_vector_tensor = torch.from_numpy(np.array(loaded[:, 1:]))
classes_column = loaded[:,0].tolist()
# normalize both the new fv and the old ones
known_feature_vectors_normalized = torch.nn.functional.normalize(feature_vector_tensor, p=2, dim=1)

# the img for the query
new_feature_vector_normalized = get_vector('output/training_pieces/0000_D5.png', model)
new_feature_vector_normalized = torch.nn.functional.normalize(new_feature_vector_normalized, p=2, dim=0)

# Compute cosine similarity between new_feature_vector and every feature vector in known_feature_vectors
similarities = torch.nn.functional.cosine_similarity(known_feature_vectors_normalized, new_feature_vector_normalized)

# Sort the indices based on cosine similarity in descending order
sorted_indices = torch.argsort(similarities, descending=True)

# Get the indices of the top 10 most similar images
top_10_indices = sorted_indices[:10]

print("classes of the 10 most similar:")
[print(inverted_classes[classes_column[index]]) for index, _ in enumerate(classes_column) if index in top_10_indices]