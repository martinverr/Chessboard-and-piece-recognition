from PIL import Image
import torch
import numpy as np
import os
from collections import Counter

def get_vector(img, model, lenght_feature_vector=2048):
    model.eval()
    # 1. Load the image with Pillow library
    t_img = img.unsqueeze(0)
    # 2. Create a PyTorch Variable with the transformed image
    #t_img = model.transform(img).unsqueeze(0)
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

def retrieval(imgs, model=None, lenght_feature_vector=2048, model_name ='2-ResNet50', model_saves_path = './scratch-cnn/modelsaves2/'):

    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(f"{model_saves_path}{model_name}.pth", map_location=device)

    loaded = np.load('retrieval/feature_vector_pieces.npy')

    feature_vector_tensor = torch.from_numpy(np.array(loaded[:, 1:].astype(float)))
    paths_column = loaded[:,0].tolist()
    # normalize both the new fv and the old ones
    known_feature_vectors_normalized = torch.nn.functional.normalize(feature_vector_tensor, p=2, dim=1)

    # the img for the query
    for img in imgs:
        new_feature_vector_normalized = get_vector(img, model)
        new_feature_vector_normalized = torch.nn.functional.normalize(new_feature_vector_normalized, p=2, dim=0)

        # Compute cosine similarity between new_feature_vector and every feature vector in known_feature_vectors
        similarities = torch.nn.functional.cosine_similarity(known_feature_vectors_normalized, new_feature_vector_normalized)

        # Sort the indices based on cosine similarity in descending order
        sorted_indices = torch.argsort(similarities, descending=True)

        # Get the indices of the top 10 most similar images
        top_10_indices = sorted_indices[:10]

        classes_of_first_10 = []
        #print("Path of the 10 most similar:")
        [classes_of_first_10.append(os.path.basename(os.path.dirname(paths_column[index]))) for index,path in enumerate(paths_column) if index in top_10_indices]
        #[print(paths_column[index], ' --> ', os.path.basename(os.path.dirname(paths_column[index]))) for index,path in enumerate(paths_column) if index in top_10_indices]

        # Conta le occorrenze di ciascuna stringa
        count = Counter(classes_of_first_10)

        # Trova l'elemento più comune
        most_frequent_class = count.most_common(1)[0][0]
        results.append(most_frequent_class)

    return results
