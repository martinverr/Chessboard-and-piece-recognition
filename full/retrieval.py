from PIL import Image
import torch
import numpy as np
import os
from collections import Counter
import time
import glob
import cv2
import torchvision.transforms as transforms
from models import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt


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


def sum_to_one_normalize(lst):
    total_sum = sum(lst)
    normalized_lst = [x / total_sum for x in lst]
    return normalized_lst

def retrieval(imgs, model=None, lenght_feature_vector=2048, model_name ='2-ResNet50', model_saves_path = './scratch-cnn/modelsaves2/', fv_path = 'retrieval/feature_vector_pieces',ensamble = False):
    # classes = {'b_Bishop' : 0, 'b_King' : 1, 'b_Knight' : 2, 'b_Pawn' : 3, 'b_Queen' : 4,
    #                  'b_Rook' : 5, 'w_Bishop' : 6, 'w_King' : 7, 'w_Knight' : 8, 'w_Pawn' : 9, 'w_Queen' : 10,
    #                  'w_Rook' : 11} resnet50
    classes = {
  'b_Pawn': 0,
  'b_Bishop': 1,
  'b_Knight': 2,
  'b_Rook': 3,
  'b_Queen': 4,
  'b_King': 5,
  'w_King': 6,
  'w_Queen': 7,
  'w_Rook': 8,
  'w_Knight': 9,
  'w_Bishop': 10,
  'w_Pawn': 11
}
    
    #class_names = ['b_Knight', 'w_Knight','w_Queen','b_Queen','b_Rook','w_Bishop','w_Rook','w_Pawn','b_King','w_King','b_Pawn','b_Bishop'] 
    class_names = ['b_Pawn','b_Bishop','b_Knight','b_Rook','b_Queen','b_King', 'w_King','w_Queen','w_Rook','w_Knight','w_Bishop','w_Pawn'] 

    inverted_classes = {value: key for key, value in classes.items()}
    results = []
    if model == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(f"{model_saves_path}{model_name}.pth", map_location=device)

    else:
        model = model

    loaded = torch.load(f'{fv_path}.pt')

    feature_vector_tensor = torch.from_numpy(np.array(loaded[:, 1:]))
    classes_column = loaded[:,0].tolist()
    # normalize both the new fv and the old ones
    known_feature_vectors_normalized = torch.nn.functional.normalize(feature_vector_tensor, p=2, dim=1)

    # the img for the query
    numb = 0
    for img in imgs:
        print(f"Computing img {numb}")
        numb = numb +1
        new_feature_vector_normalized = get_vector(img, model, lenght_feature_vector = lenght_feature_vector)
        new_feature_vector_normalized = torch.nn.functional.normalize(new_feature_vector_normalized, p=2, dim=0)

        # Compute cosine similarity between new_feature_vector and every feature vector in known_feature_vectors
        similarities = torch.nn.functional.cosine_similarity(known_feature_vectors_normalized, new_feature_vector_normalized)

        # Sort the indices based on cosine similarity in descending order
        sorted_indices = torch.argsort(similarities, descending=True)

        # Get the indices of the top 10 most similar images
        top_10_indices = sorted_indices[:10]

        classes_of_first_10 = []
        #print("classes of the 10 most similar:")
        #[print(inverted_classes[classes_column[index]]) for index, _ in enumerate(classes_column) if index in top_10_indices]
        [classes_of_first_10.append(inverted_classes[classes_column[index]]) for index, _ in enumerate(classes_column) if index in top_10_indices]


        # Conta le occorrenze di ciascuna stringa
        count = Counter(classes_of_first_10)

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

if __name__ == "__main__":
    model2_name = "ResNet18_80x160"
    print(f"Retrieval test - {model2_name}")
    img_paths = glob.glob('./test_pieces/**.png')
    annotation_path = glob.glob('./test_pieces/**.txt')
    class_names = ['b_Knight', 'w_Knight','w_Queen','b_Queen','b_Rook','w_Bishop','w_Rook','w_Pawn','b_King','w_King','b_Pawn','b_Bishop'] 
    correct = 0
    accuracy = 0


    if len(img_paths) == len(annotation_path):
        image_tensors = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model2 = torch.load(f"./scratch-cnn/modelsaves3/{model2_name}.pth", map_location=device)

        for path in img_paths:
            image = cv2.imread(path)  # Open image
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            pieces_cnn_input = model2.transform(img).reshape([3, 80, 160])
            image_tensors.append(pieces_cnn_input)  # Add tensor to the list

        ground_true = []
        for ann in annotation_path:
                with open(ann, 'r') as file:
                    # Read the first line
                    first_line = file.readline().strip()
                    # Extract the first word from the first line
                    if first_line:
                        first_word = first_line.split()[0]
                        ground_true.append(first_word)
        
        
        start_time = time.time()

        result_retrieval = retrieval(image_tensors, model=model2, lenght_feature_vector=512, fv_path = 'retrieval/feature_vector_pieces_resnet18' )

        end_time = time.time()

        for ret,lable in zip(result_retrieval, ground_true):
            if ret == lable:
                correct += 1
                print(f"{ret} | {lable}")
            else:
                print(ret, lable)

        duration = end_time - start_time
        print(f"The entire operation took {duration} seconds.")


        # Calcolo della matrice di confusione
        cm = confusion_matrix(ground_true, result_retrieval, labels= class_names)
        cmd = ConfusionMatrixDisplay(cm, display_labels=class_names)

        # Plot della matrice di confusione
        fig, ax = plt.subplots(figsize=(8, 8))
        cmd.plot(ax=ax, cmap="Blues")
        plt.xticks(rotation=90)
        plt.title("Confusion Matrix")
        plt.show()

        # Calcola l'accuracy
        accuracy = accuracy_score(ground_true, result_retrieval)

        # Calcola l'F1 score
        f1 = f1_score(ground_true, result_retrieval, average='macro')

        # Calcola la precision
        precision = precision_score(ground_true, result_retrieval, average='macro')

        # Calcola la recall
        recall = recall_score(ground_true, result_retrieval, average='macro')

        print(f'Accuracy: {accuracy:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
    else:
        raise TypeError("Error in lengh directory, missmatch annotation and img lenght")