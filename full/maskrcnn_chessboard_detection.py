import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from models import *
import matplotlib.pyplot as plt


def get_board_with_maskrcnn(image_path, model = None, verbose_show = False):

    '''
        return: 4 board points
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model == None:
        model = MaskRCNN_board() 
        model.model.load_state_dict(torch.load('./maskRCNN_epoch_4_.pth', map_location = device))
        model.to(device)
        model.eval()


    img = Image.open(image_path)
    img_bg = cv2.imread(image_path)
    img = model.transform(img).unsqueeze(0)

    with torch.no_grad():
        img = img.to(device)
        pred = model(img)

    
    if verbose_show:
        cv2.imshow('mask', (pred[0]["masks"][0].cpu().detach().numpy() * 255).astype("uint8").squeeze())
        #cv2.imwrite('./out_maskrcnn.jpg', (pred[0]["masks"][0].cpu().detach().numpy() * 255).astype("uint8").squeeze())
        cv2.waitKey(0)


    board = np.where(pred[0]['masks'].cpu().numpy() > 0.4, 255, 0).astype('uint8').squeeze()
    if verbose_show:
        cv2.imshow('thresholded', board)
        #cv2.imwrite('./thresholded_mask.jpg', board)
        colored_mask = np.stack((board,) * 3, axis=-1)
            # Crea una maschera booleana per i valori dove board è 255.
        mask = board == 255
    
        # Applica la maschera a ogni canale BGR.
        colored_mask[mask, 0] = 0  # Canale blu
        colored_mask[mask, 1] = 0  # Canale verde
        colored_mask[mask, 2] = 255  # Canale rosso
        alpha = 0.5
        overlayed_img = cv2.addWeighted(img_bg, 1, colored_mask, alpha, 0)
        #cv2.imwrite('./overlayed_mask.jpg', overlayed_img)
        cv2.waitKey(0)

    board = np.asarray(board).astype(np.uint8)

    contours, _ = cv2.findContours(board, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour

    contour_image = np.zeros_like(img_bg)

    cv2.drawContours(contour_image, [largest_contour], -1, 255, 2)

    if verbose_show:
        cv2.imshow('contour', contour_image)
        
        cv2.waitKey()

    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approximated_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)


    if verbose_show:
        cv2.polylines(overlayed_img, [approximated_polygon], True, 255, 2)
        for point in approximated_polygon:
            x, y = point[0]  # Estrai le coordinate x e y del punto
            cv2.circle(overlayed_img, (x, y), 10, (0, 255, 0), 5)  # Disegna un cerchio rosso con raggio 5
        cv2.imshow('board detected', overlayed_img)
        #cv2.imwrite('./board_detected.jpg', overlayed_img)
        cv2.waitKey()

    # if verbose_show:
    #     img_copy = cv2.imread(image_path)
    #     for point in approximated_polygon:
    #         img_c = img_copy.copy()
    #         cv2.circle(img_c, point[0], 5, (0, 255, 0), -1)
    #         cv2.imshow(f'board {point}', img_c)
    #         cv2.waitKey()

    if len(approximated_polygon) != 4:
        return None
    return approximated_polygon