from chessboard_detection import *
import torch
import torchvision.transforms as transforms
import os, glob
import json
from PIL import Image

def bbox_annotation(annotation_path) -> list:
    ## TODO: convert in tensor for better memory qol
    list_bbox_gt = []
    with open(annotation_path, 'r') as f:
        data_json = json.load(f)
        for piece in data_json["pieces"]:
            # an array of 4 numbers in the json file
            list_bbox_gt. append(piece["box"])
    return list_bbox_gt

def transformation_bboxgt(list_true_bbox, matrix) -> list:
    x,y,w,h = list_true_bbox
    original_points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    o_points = original_points.reshape(-1,1,2).astype(np.float32)
    # Apply perspective transformation to the bbox corners
    warped_points = cv2.perspectiveTransform(o_points, matrix)
    # Calculate the bounding box for the warped image
    x_min = int(np.min(warped_points[:, :, 0]))
    y_min = int(np.min(warped_points[:, :, 1]))
    x_max = int(np.max(warped_points[:, :, 0]))
    y_max = int(np.max(warped_points[:, :, 1]))
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def main():
    ## import all the img to compute
    input_imgs = glob.glob('./input/**.png')
    input_annotations = glob.glob('./input/**.json')
    print(input_imgs)

    for input_img, input_annotation in zip(input_imgs, input_annotations):
        list_bboxgt_img_tranf = []
        if not os.path.isfile(input_img) or not os.path.isfile(input_annotation):
            continue
        
        if not input_img.lower().endswith(".png") and not input_annotation.lower().endswith(".json"):
            continue
        
        print(f"image and json file found for file: {os.path.basename(input_img).strip('.png')}")

        imgname = input_img.split('\\')[-1]
    
        ## compute the warp of the img
        warpedBoardImg, matrix = board_detection(input_img, 
                                         f"{'output_' + imgname}",
                                         verbose_show=False, 
                                         verbose_output=False)
        # if no warp is found skip and also throw away the related matrix
        if warpedBoardImg is None and matrix is None:
            print(f"An error has occured so the file {os.path.basename(input_img).strip('.png')} is skipped")
            continue

        # read json bbox values - list of every bboxgt - 
        list_bboxgt_img = bbox_annotation(input_annotation)
        ## use the matrix to converte the pixel
        ### ATTENZIONE POSSIBILI VALORI NEGATIVI DELLE BBOX
        [list_bboxgt_img_tranf.append(transformation_bboxgt(bbox_gt,matrix)) for bbox_gt in list_bboxgt_img]


        verbose = True
        if verbose:
            original_img = cv2.imread(input_img)
            warped_img = warpedBoardImg.copy()
            for gt, ngt in zip(list_bboxgt_img, list_bboxgt_img_tranf):
                x,y,w,h = gt
                x_min,y_min,w_max ,h_max = ngt
                print(x,y,x+w,y+h)
                print(x_min,y_min,x_min+w_max,y_min +h_max)
                cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(warped_img, (x_min, y_min), (x_min + w_max, y_min + h_max), (0, 255, 0), 2)

            # Display the original and warped images with bounding boxes
            cv2.imshow('Original Image with Bbox', original_img)
            cv2.imshow('Warped Image with Bbox', warped_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        
        ## given 4 point cut the img
        

if __name__ == "__main__":
    main()