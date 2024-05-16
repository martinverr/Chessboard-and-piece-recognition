from chessboard_detection import *
import FEN
import torch
import torchvision.transforms as transforms
import os, glob
import json
from PIL import Image

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

def read_json_annotation(annotation_path) -> list[list, list, list, str]:
    """Read the JSON annotation file to retrieve bounding box values, piece types, and positions on the chessboard.

    Args:
        annotation_path (str): annotation path

    Returns:
        list[list, list, list, str]: list of bbox, list of piece type and list of positions | view of the chessboard
    """
    ## TODO: convert in tensor for better memory qol
    list_bbox_gt = []
    list_piece_type = []
    list_square = []
    turn = ""
    with open(annotation_path, 'r') as f:
        data_json = json.load(f)
        turn_bool = data_json['white_turn']
        for piece in data_json["pieces"]:
            # an array of 4 numbers in the json file
            list_bbox_gt. append(piece["box"])
            list_piece_type.append(pice_conversion(piece["piece"]))
            list_square.append(piece["square"])

        if turn_bool:
            turn = "white"
        else:
            turn = "black"

    return list_bbox_gt, list_piece_type, list_square, turn

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
    file_path_annotation = 'json_annotation_gt_pieces.json'
    file_path_annotation_custom =  'json_annotation_custom_pieces.json'
    sample = {}
    custom_sample = {}
    list_not_found = []
    #print(input_imgs)

    for input_img, input_annotation in zip(input_imgs, input_annotations):
        list_bboxgt_img_tranf = []
        json_list = []
        custom_json_list = []

        if not os.path.isfile(input_img) or not os.path.isfile(input_annotation):
            continue
        
        if not input_img.lower().endswith(".png") and not input_annotation.lower().endswith(".json"):
            continue
        
        print(f"image and json file found for file: {os.path.basename(input_img).strip('.png')}")

        imgname = input_img.split('\\')[-1]
    
        ## compute the warp of the img
        try:
            warpedBoardImg, matrix = board_detection(input_img, 
                                            f"{'output_' + imgname}",
                                            verbose_show=False, 
                                            verbose_output=False)
            # if no warp is found skip
            if warpedBoardImg is None and matrix is None:
                print(f"An error has occured so the file {os.path.basename(input_img).strip('.png')} is skipped")
                list_not_found.append(imgname)
                continue


            # read json bbox values - list of every bboxgt - list of piece (not used), list of square position (used in the indexing of the crop) 
            list_bboxgt_img, list_pice_img, list_square_img, view = read_json_annotation(input_annotation)
            
            square_info = grid_detection(warpedBoardImg, view)
            
            if square_info is None:
                print(f"An error has occured so the file {os.path.basename(input_img).strip('.png')} is skipped")
                list_not_found.append(imgname)
                continue

            index_custom_bbox = [num for num in range(len(square_info[:,1])) if square_info[num,1].lower() in list_square_img]
            for indx in index_custom_bbox:
                custom_bbox = square_info[indx,-1]
                c_x,c_y,c_w,c_h = custom_bbox
                custom_piece = list_pice_img[list_square_img.index(square_info[indx,1].lower())]
                custom_position = square_info[indx,1].lower()
                #json_list.append({"piece": str(piece), "position": str(square), "bbox": [x,y,w,h]})
                custom_json_list.append({"piece": str(custom_piece), "position": str(custom_position), "bbox": [int(c_x),int(c_y),int(c_w),int(c_h)]})


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

            for bbox, square, piece in zip(list_bboxgt_img_tranf, list_square_img, list_pice_img):
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
                json_list.append({"piece": str(piece), "position": str(square), "bbox": [x,y,w,h]})
                if verbose:
                    cv2.imshow(f'Singular bbox for {square} in img {input_img}', new_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                
                ## save the new img in path '/output/bboxgt/{index_name}_{position}.png'
                os.makedirs(dir_save_bbox_gt,exist_ok=True)
                #cv2.imwrite(os.path.join(dir_save_bbox_gt, f'{imgname.strip(".png")}_{square.upper()}.png'), new_img)

            ## create json file with the following dict -> [{"{index_img}": [{"piece": k_black, "position": h8, "bbox": [x,y,w,h]}, {"piece":....}]]}
            sample[f"{imgname.strip('.png')}"] = json_list
            custom_sample[f"{imgname.strip('.png')}"] = custom_json_list
        except:
            print(f"error {imgname}")

    print(list_not_found)
    with open("img_not_found.txt", 'w') as fp:
        fp.write(str(list_not_found))

    # with open(file_path_annotation, 'w') as fp:
    #     json.dump(sample, fp)

    # with open(file_path_annotation_custom, 'w') as fp:
    #     json.dump(custom_sample, fp)
    
            

def from_json_to_annotation(file_path_annotation, index, mask = False):
    list_values =[]
    with open(file_path_annotation, 'r') as f:
        data_json = json.load(f)

    if index in data_json:
        # Iterate through the list of dictionaries for the specified image index
        for piece_info in data_json[index]:
            # Access the values using dictionary keys
            piece = piece_info["piece"]
            position = piece_info["position"]
            bbox = piece_info["bbox"]
            if mask:
                mask = torch.zeros(800,700, dtype=bool)
                x,y,w,h = bbox
                mask[y:y+h,x:x+w] = 1
                # Process the piece information as needed
                #print(f"Piece: {piece}, Position: {position}, Bbox: {bbox}")
                list_values.append({"piece": piece, "position": position, "mask": mask})
            else:
                list_values.append({"piece": piece, "position": position, "bbox": bbox})

        return {f"{index}": list_values}
    else:
        print(f"Image index '{index}' not found in the JSON data.")
        return None
    

        

if __name__ == "__main__":
    file_path_annotation = 'json_annotation_gt_pieces.json'
    file_path_annotation_custom =  'json_annotation_custom_pieces.json'
    main()
    #print(from_json_to_annotation(file_path_annotation_custom, "0024"))