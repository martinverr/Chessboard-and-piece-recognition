from PIL import Image
import torch
import numpy as np
import os
from collections import Counter

def get_vector(img, model, lenght_feature_vector=512):
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

def sum_to_one_normalize(lst):
    total_sum = sum(lst)
    normalized_lst = [x / total_sum for x in lst]
    return normalized_lst

def retrieval(imgs, model=None, lenght_feature_vector=512, model_name ='ResNet18_80x160', model_saves_path = './scratch-cnn/modelsaves2/', ensamble=False):
    classes = {'b_Pawn': 0, 'b_Bishop': 1, 'b_Knight': 2, 'b_Rook': 3, 'b_Queen': 4, 'b_King': 5, 'w_King': 6, 'w_Queen': 7, 'w_Rook': 8, 'w_Knight': 9, 'w_Bishop': 10, 'w_Pawn': 11}
    class_names = ['b_Pawn','b_Bishop','b_Knight','b_Rook','b_Queen','b_King', 'w_King','w_Queen','w_Rook','w_Knight','w_Bishop','w_Pawn']
    
    inverted_classes = {value: key for key, value in classes.items()}
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(f"{model_saves_path}{model_name}.pth", map_location=device)

    loaded = torch.load('retrieval/feature_vector_pieces_resnet18.pt')

    feature_vector_tensor = torch.from_numpy(np.array(loaded[:, 1:]))
    classes_column = loaded[:,0].tolist()
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
        top_20_indices = sorted_indices[:20]

        classes_of_first_20 = []
        #print("classes of the 10 most similar:")
        #[print(inverted_classes[classes_column[index]]) for index, _ in enumerate(classes_column) if index in top_10_indices]
        [classes_of_first_20.append(inverted_classes[classes_column[index]]) for index, _ in enumerate(classes_column) if index in top_20_indices]

        # Conta le occorrenze di ciascuna stringa
        count = Counter(classes_of_first_20)

        if ensamble:
            # Trova la probability distribution+
            list = []
            for k in class_names:
                if k in count.keys():
                    list.append(count[k])
                else:
                    list.append(0)
            normalize_probability = sum_to_one_normalize(list)
            results.append(normalize_probability)
        else:
            # Trova l'elemento più comune
            most_frequent_class = count.most_common(1)[0][0]
            results.append(most_frequent_class)


    return results
