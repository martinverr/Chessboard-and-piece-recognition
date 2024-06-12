import torch
from chessboard_detection import *
from models import *
import json
import numpy as np
from skimage.draw import polygon

HEIGHT, WIDTH = 800, 1200 

def calculate_iou(mask_one:torch.Tensor, mask_two: torch.Tensor ) -> torch.Tensor :
    intersection = torch.logical_and(mask_one,mask_two)
    union = torch.logical_or(mask_one,mask_two)
    #np.sum(intersection) / np.sum(union)
    iou = torch.sum(intersection)/torch.sum(union)
    return iou

def read_json_annotation_bbox(annotation_path) -> list:
    """Read the JSON annotation file to retrieve bounding box value of the chessboard.

    Args:
        annotation_path (str): annotation path

    Returns:
        list: list of bbox of the chessborad
    """
    ## TODO: convert in tensor for better memory qol
    list_bbox_chessboard = []
    with open(annotation_path, 'r') as f:
        data_json = json.load(f)
        list_bbox_chessboard = data_json['corners']

    if 0 == len(list_bbox_chessboard):
        raise TypeError("Empty ground true corner of the chesboard")

    return list_bbox_chessboard

# def mask_poligon(p1, p2, p3, p4):


if __name__ == "__main__":
    input_imgs = glob.glob('./test/**.png')
    json_img = glob.glob('./test/**.json')
    print(input_imgs)

    threshold = 0.8
    true_positive = []
    total = len(input_imgs)


    for img_png, json_png in zip(input_imgs, json_img):
        print(f"Currently computing image number: {img_png}")
        mask_predicted = torch.zeros((HEIGHT, WIDTH), dtype=bool)
        mask_gt = torch.zeros((HEIGHT, WIDTH), dtype=bool)
        predicted_bbox = get_board_with_maskrcnn(img_png)
        gt_bbox = read_json_annotation_bbox(json_png)

        if gt_bbox is None or predicted_bbox is None:
            print("Error with finding the bbox")
            continue
        elif len(gt_bbox) == 0 or predicted_bbox.size == 0:
            print("One of the arrays is empty")
            continue

        predicted_bbox = [item[0] for item in predicted_bbox]
        predicted_bbox = np.array(predicted_bbox)
        r_p, c_p = polygon(predicted_bbox[:, 1], predicted_bbox[:, 0], mask_predicted.shape)

        gt_bbox = np.array(gt_bbox)
        r_gt, c1_gt = polygon(gt_bbox[:, 1], gt_bbox[:, 0], mask_gt.shape)

        #generate the mask poligon
        mask_predicted[r_p, c_p] = True
        mask_gt[r_gt, c1_gt] = True
        
        iou = calculate_iou(mask_predicted, mask_gt)
        if iou > threshold:
            true_positive.append(iou)

    # calculate medan iou
    mean_iou = sum(true_positive) / len(true_positive)
    print(f"Mean iou: {mean_iou}")
    print(f"Number of positive with threhold {threshold}: {len(true_positive)}")
    print(f"Total number of samples: {total}")

        
        


