from chessboard_detection import *
import torch
import torchvision.transforms as transforms
import os, glob
import json
from PIL import Image
from FEN import FEN

def pice_conversion(lable) -> str:
    """Convert the piece type to our standard label name.

    Args:
        lable (str): key value of the internal dictionary

    Returns:
        str: our version of the lable name
    """
    lable_to_convention = {
                'K': 'w_King', 'Q': 'w_Queen', 'R': 'w_Rook', 'B': 'w_Bishop', 'N': 'w_Knight', 'P': 'w_Pawn',
                'k': 'b_King', 'q': 'b_Queen', 'r': 'b_Rook', 'b': 'b_Bishop', 'n': 'b_Knight', 'p': 'b_Pawn'
            }
    return lable_to_convention[lable]

def read_json_annotation(annotation_path) -> list[list, list, list]:
    """Read the JSON annotation file to retrieve bounding box values, piece types, and positions on the chessboard.

    Args:
        annotation_path (str): annotation path

    Returns:
        list[list, list, list]: list of bbox, list of piece type and list of positions
    """
    ## TODO: convert in tensor for better memory qol
    list_bbox_gt = []
    list_piece_type = []
    list_square = []
    with open(annotation_path, 'r') as f:
        data_json = json.load(f)
        for piece in data_json["pieces"]:
            # an array of 4 numbers in the json file
            list_bbox_gt. append(piece["box"])
            list_piece_type.append(pice_conversion(piece["piece"]))
            list_square.append(piece["square"])
    return list_bbox_gt, list_piece_type, list_square

def transformation_bboxgt(list_true_bbox, matrix) -> list:
    """Given a transformation matrix and the coordinates of the bounding box (bbox) of the original image, project the points onto the new warped image.

    Args:
        list_true_bbox (list): bbox of the true img
        matrix (Tensor): transformation matrix found in the warping part

    Returns:
        list: value of the projected points from the true bbox
    """
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
    dir_save_bbox_gt = './output/bboxgt'
    #print(input_imgs)

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
        # if no warp is found skip
        if warpedBoardImg is None and matrix is None:
            print(f"An error has occured so the file {os.path.basename(input_img).strip('.png')} is skipped")
            continue

        # read json bbox values - list of every bboxgt - list of piece (not used), list of square position (used in the indexing of the crop) 
        list_bboxgt_img, _, list_square_img = read_json_annotation(input_annotation)
        ## use the matrix to converte the pixel
        ### ATTENZIONE POSSIBILI VALORI NEGATIVI DELLE BBOX -> gestito sotto quando faccio il crop delle img
        [list_bboxgt_img_tranf.append(transformation_bboxgt(bbox_gt,matrix)) for bbox_gt in list_bboxgt_img]

        # for debugging purpose
        verbose = False
        if verbose:
            original_img = cv2.imread(input_img)
            warped_img = warpedBoardImg.copy()
            for gt, ngt in zip(list_bboxgt_img, list_bboxgt_img_tranf):
                x,y,w,h = gt
                x_min,y_min,w_max ,h_max = ngt
                cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(warped_img, (x_min, y_min), (x_min + w_max, y_min + h_max), (0, 255, 0), 2)

            # Display the original and warped images with bounding boxes
            cv2.imshow('Original Image with Bbox', original_img)
            cv2.imshow('Warped Image with Bbox', warped_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        ## given 4 point cut the img

        for bbox, square in zip(list_bboxgt_img_tranf, list_square_img):
            x,y,w,h = bbox
            yh = y+h
            xw =x+w
            # control if the coordinates are out of bound
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if yh > warpedBoardImg.shape[0]:
                yh = warpedBoardImg.shape[0]
            if xw > warpedBoardImg.shape[1]:
                xw = warpedBoardImg.shape[1]

            # print per il debug
            # print(x,y,yh,xw, (800-(yh-y)) + (yh-y), (700-(xw-x)) + (xw-x))
            new_img = warpedBoardImg[y:yh,x:xw]
            if verbose:
                cv2.imshow(f'Singular bbox for {square} in img {input_img}', new_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            ## save the new img in path '/output/bboxgt/{index_name}_{position}.png'
            os.makedirs(dir_save_bbox_gt,exist_ok=True)
            cv2.imwrite(os.path.join(dir_save_bbox_gt, f'{imgname.strip(".png")}_{square.upper()}.png'), new_img)

        

if __name__ == "__main__":
    main()