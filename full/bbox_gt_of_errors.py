import ast
import json
import os
from FEN import FEN
import numpy as np
import cv2
import glob
#TODO as generate featurevector qquando separo in altra cartella

def parseJSONpieces(filename):
    with open(filename, 'r') as jsonfile:
        jsondata = json.load(jsonfile)
    return jsondata['pieces'], jsondata['corners'], jsondata['white_turn']


def generate_bbox(input_img_paths, input_annotations_paths, dst_dir_path, verbose=False):
    for input_img_path, input_annotation_path in zip(input_img_paths, input_annotations_paths):
        print(f"{input_img_path}")
        json_pieces, json_corners, white_turn = parseJSONpieces(input_annotation_path)
        
        # Get perspectrive matrix using the 4 corners and the transf img
        if white_turn:
            pts1 = np.float32([
            [json_corners[1][0], json_corners[1][1]],
            [json_corners[2][0], json_corners[2][1]],
            [json_corners[0][0], json_corners[0][1]], 
            [json_corners[3][0], json_corners[3][1]],
            ])
        else:
            pts1 = np.float32([
            [json_corners[3][0], json_corners[3][1]],
            [json_corners[0][0], json_corners[0][1]],
            [json_corners[2][0], json_corners[2][1]],
            [json_corners[1][0], json_corners[1][1]],
            ])
        pts2 = np.float32([[50, 150], [650, 150], [50, 850], [650, 850]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        newimg = cv2.warpPerspective(cv2.imread(input_img_path), matrix, (700, 850))
    
        for piece in json_pieces:
            x,y,w,h = piece['box']
            
            piece["piece"] = FEN.fen_to_piece[piece['piece']]
            piece["square"] = piece['square'].upper()
            piece['box'] = [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
            
            original_points = np.array(piece["box"])
            original_points = original_points.reshape(-1,1,2).astype(np.float32)

            new_points_transformed = cv2.perspectiveTransform(original_points, matrix)
            
            x_min = int(np.min(new_points_transformed[:, :, 0]))
            y_min = int(np.min(new_points_transformed[:, :, 1]))
            x_max = int(np.max(new_points_transformed[:, :, 0]))
            y_max = int(np.max(new_points_transformed[:, :, 1]))
            x, y, xw, yh = [max(0, x_min), max(0, y_min), min(newimg.shape[1], x_max), min(newimg.shape[0], y_max)]

            piecetransformed = newimg[y:yh,x:xw]
            
            if verbose:
                cv2.imshow(f'{piece["square"]}', piecetransformed)
                cv2.waitKey(0)
            cv2.destroyAllWindows()

            #save img
            imgnumber = os.path.splitext(input_img_path)[0].split('\\')[-1]        
            cv2.imwrite(os.path.join(dst_dir_path, f'{imgnumber}_{piece["square"].upper()}.png'), piecetransformed)
            #save txt class
            with open(os.path.join(dst_dir_path, f'{imgnumber}_{piece["square"].upper()}.txt'), 'w') as f:
                f.write(piece['piece'])


def main():
    dst_dir_path = './output/error_pieces_gt/'
    os.makedirs(os.path.dirname(dst_dir_path), exist_ok=True)

    error_list = []
    if os.path.isfile("./full/errors.txt"):
        with open("./full/errors.txt", "r") as file:
            error_list = ast.literal_eval(file.read())
            error_list.sort()

    input_annotations_paths = [path for path in glob.glob('./input/**.json')
                    if os.path.splitext(path)[0].split('\\')[-1] in error_list]
    input_img_paths = [path for path in glob.glob('./input/**.png')
                    if os.path.splitext(path)[0].split('\\')[-1] in error_list]
    
    generate_bbox(input_img_paths, input_annotations_paths, dst_dir_path)

if __name__ == "__main__":
    main()