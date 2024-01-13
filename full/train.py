import os, glob
from FEN import FEN
from chessboard_detection import *


def classify():
    return {}


def main():
    input_imgs = glob.glob('./input/**')
    print(input_imgs)

    for input_img in input_imgs:
        if not os.path.isfile(input_img):
            continue
        
        if not input_img.lower().endswith(".png"):
            continue
        
        print(f"image file found: {input_img}")

        if not os.path.isfile(os.path.splitext(input_img)[0] + '.json'):
            print("not found related json")
            continue

        imgname = input_img.split('\\')[-1]
    
        warpedBoardImg = board_detection(input_img, 
                                         f"{'output_' + imgname}",
                                         verbose_show=False, 
                                         verbose_output=False)
        if warpedBoardImg is None:
            continue

        
        truth = FEN(os.path.splitext(input_img)[0])
        true_fen, true_pos, viewpoint = truth.fen, truth.pieces, truth.view
        #debug lettura json
        if False:
            print("\n##### DEBUG JSON START ######\n")
            print(f"FEN: {true_fen}\n\nPieces Position: {true_pos}\n\nViewpoint: {viewpoint}")
            print("\n##### DEBUG JSON END ######\n")

        grid_squares = grid_detection(warpedBoardImg,
                                 viewpoint,
                                 verbose_show=False)
        if grid_squares is None:
            continue

        
        # Extend the information to include piece information in 3rd col (image remain last in 4th col)
        grid_squares = np.column_stack((grid_squares[:,:2], 
                                         [true_pos.get(coord, 'empty') for coord in grid_squares[:, 1]],
                                         grid_squares[:,-2:]
                                         ))
        """
        numpy.ndarray(64, 4):
        64 squares of [square num, coord, piece, img]
        example: 
            0: [0, 'h1', 'empty', array([...], uint8)]
            1: [1, 'g1', 'b_Rook', array([...], uint8)]
            ...
            63: [63, 'a8', 'w_Rook', array([...], uint8)]
        """
        print(f'obtained data for training')
        predicted_pos = classify()
        

if __name__ == "__main__":
    main()