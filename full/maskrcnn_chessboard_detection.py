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
        model.model.load_state_dict(torch.load('./maskRCNN_epoch_2_.pth', map_location = device))

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
        cv2.waitKey(0)


    board = np.where(pred[0]['masks'].cpu().numpy() > 0.4, 255, 0).astype('uint8').squeeze()
    if verbose_show:
        cv2.imshow('thresholded', board)
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
    cv2.polylines(img_bg, [approximated_polygon], True, 255, 2)

    if verbose_show:
        cv2.imshow('board detected', img_bg)
        cv2.waitKey()

    return approximated_polygon