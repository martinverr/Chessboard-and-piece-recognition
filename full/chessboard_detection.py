import numpy as np
from time import time
import cv2

from ChessLinesClustering import ChessLines
from chessboard_detection_functions import *

from maskrcnn_chessboard_detection import *

import warnings
warnings.filterwarnings("ignore")

def _board_detection_geometic(fname : str, verbose_show=False):
    """
    Given a filename, returns the warped board image.

    Parameters
    ----------
    fname : String
        File img path
    verbose_show : Set true to show intermidiate steps
        default False
        
    Return
    ------
    np.ndarray
    Image warped 700x800 
    """

    img = cv2.imread(fname)
    assert img is not None
    img_for_wrap = np.copy(img)
    if verbose_show:
        cv2.imshow("img", img)
        cv2.waitKey(0)

    bilateral_img = img.copy()
    bilateral_img = cv2.bilateralFilter(img, 3, 100, 70)
    if verbose_show:
        cv2.imshow(f"bilateral filter", bilateral_img)
        cv2.waitKey(0)

    gray = cv2.cvtColor(bilateral_img, cv2.COLOR_BGR2GRAY)


    #TODO adaptive threshold
    edges = cv2.Canny(gray, 70, 400, apertureSize=3)
    if verbose_show:
        cv2.imshow(f"cannied", edges)
        cv2.waitKey(0)

    # Hough line prob detection
    lines2 = cv2.HoughLinesP(edges, 1, np.pi/180, 69, minLineLength=30, maxLineGap=50)
    lines2 = np.reshape(lines2, (-1, 4))
    lines2 = np.array([two_points_to_polar(line) for line in lines2])
    
    if verbose_show:
        imgcopy2 = img.copy()     
        output_lines(imgcopy2, lines2, (0,255,0))
        cv2.imshow("lines prob", imgcopy2)
        cv2.waitKey(0)
    
    
    W, H = img.shape[0] , img.shape[1]
    chessLines = ChessLines(lines2, W, H)
    
    # Horizontal and Vertical lines
    hLines = chessLines.getHLines()
    vLines = chessLines.getVLines()

    # Pulizia linee
    hLines, hLinesRemoved = removeOutLiers(hLines)
    vLines, vLinesRemoved = removeOutLiers(vLines)
    chessLines.setHLines(hLines)
    chessLines.setVLines(vLines)
    
    if verbose_show:
        imgcopy = img.copy()
        output_lines(imgcopy, hLines, (0,0,255))
        output_lines(imgcopy, vLines, (255,0,255))
        output_lines(imgcopy, hLinesRemoved, (0,255,0))
        output_lines(imgcopy, vLinesRemoved, (0,255,0))
        cv2.imshow("lines standard con eliminazione outliers", imgcopy)
        cv2.waitKey(0)

    
    # clustering linee con DBSCAN
    chessLines.cluster('DBSCAN')
    hLinesCLustered = chessLines.getHLinesClustered()
    vLinesCLustered = chessLines.getVLinesClustered()

    if verbose_show:
        imgcopy = img.copy()
        output_lines(imgcopy, hLinesCLustered , (255,0,0))
        output_lines(imgcopy, vLinesCLustered, (0,255,0))
        if len(hLinesCLustered) < 9 or len(vLinesCLustered) < 9:
            cv2.putText(imgcopy,
                        "Not enough line: DISCARDED",
                        (100,100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,0,255), 2)
            cv2.putText(imgcopy, 
                        "Provide a better image",
                        (100,150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,0,255), 2)
        for line in hLinesCLustered:
            cv2.putText(imgcopy,
                        f"{line[3]}",
                        (300,int(line[3])-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255,0,0), 2)
        for line in vLinesCLustered:
            cv2.putText(imgcopy,
                        f"{int(line[2])}",
                        (int(line[2]), 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0,0,255), 2)
        cv2.imshow("grid_detection: clustered lines", imgcopy)
        cv2.waitKey(0)
            
    
    # abort if clustered lines are less than expected
    if len(hLinesCLustered) < 9 or len(vLinesCLustered) < 9:
            print("Not enough lines found, provide a better image")
            return None
    
    # 4 corner della sezione dell'immagine da warpare
    warpingSectionCorners = warpingSection(chessLines, old_version=True, margins=[-70,20,-20,20])

    # Perspective transform
    new_img = four_point_transform(img_for_wrap, warpingSectionCorners, (700, 700 + 100))
    
    if verbose_show:
        cv2.imshow("warped", new_img)
        cv2.waitKey(0)

    # return warped image of chessboard + margin
    return new_img

def board_detection(fname : str, old_version = False, verbose_show=False, model = None):
    if old_version:
        warp_img  = _board_detection_geometic(fname=fname, verbose_show=verbose_show)
    else:
        warp_img = _board_detection_maskrcnn(fname=fname, verbose_show=verbose_show, model=model)
    return warp_img
        
def _board_detection_maskrcnn(fname : str, verbose_show=False, model = None):
    points = get_board_with_maskrcnn(fname, verbose_show=verbose_show, model=model)
    if points is None:
        return None
    # Flatten the array to make sorting easier
    points_flat = points.reshape(-1, 2)

    # Sort the points by x-coordinate
    sorted_points_v = points_flat[np.argsort(points_flat[:, 0])]
    # Sort the points by y-coordinate
    sorted_points_h = points_flat[np.argsort(points_flat[:, 1])]

    # h_lines = []
    # v_lines = []
    # all_lines = []
    # for i in range(0, len(sorted_points_h)-1,2):
    #     poipoi = np.concatenate((sorted_points_h[i],sorted_points_h[i+1]), axis=0)
    #     h_line = two_points_to_polar(poipoi, verbose=False)
    #     h_lines.append(h_line)
    #     puipui = np.concatenate((sorted_points_v[i],sorted_points_v[i+1]), axis=0)
    #     v_line = two_points_to_polar(puipui, verbose=False)
    #     v_lines.append(v_line)

    #     all_lines.append(h_line)
    #     all_lines.append(v_line)

    # Initialize empty NumPy arrays
    h_lines = np.empty((0, 2))
    v_lines = np.empty((0, 2))
    all_lines = np.empty((0, 2))

    for i in range(0, len(sorted_points_h)-1, 2):
        concatenate_h = np.concatenate((sorted_points_h[i], sorted_points_h[i+1]), axis=0)
        h_line = two_points_to_polar(concatenate_h, verbose=False)
        h_lines = np.append(h_lines, [h_line], axis=0)
        
        concatenate_v = np.concatenate((sorted_points_v[i], sorted_points_v[i+1]), axis=0)
        v_line = two_points_to_polar(concatenate_v, verbose=False)
        v_lines = np.append(v_lines, [v_line], axis=0)

        all_lines = np.append(all_lines, [h_line, v_line], axis=0)


    if verbose_show:
        imgcopy = cv2.imread(fname)
        output_lines(imgcopy, h_lines , (255,0,0))
        output_lines(imgcopy, v_lines, (0,255,0))
        cv2.imshow("grid_detection: board lines from maskrcnn", imgcopy)
        cv2.waitKey(0)

    img = cv2.imread(fname)
    W, H = img.shape[0] , img.shape[1]
    chessLines = ChessLines(all_lines, W, H)
    Lines = chessLines.getHLines()
    vLines = chessLines.getVLines()
    
    if verbose_show:
        imgcopy = cv2.imread(fname)
        output_lines(imgcopy, Lines , (255,0,0))
        output_lines(imgcopy, vLines, (0,255,0))
        cv2.imshow("grid_detection: board lines from maskrcnn", imgcopy)
        cv2.waitKey(0)
    
    img_for_wrap = cv2.imread(fname)
    warpingSectionCorners = warpingSection(chessLines, margins= [-70, 50, -50, 50])
    new_img = four_point_transform(img_for_wrap, warpingSectionCorners, (700, 700 + 100))

    if verbose_show:
        cv2.imshow("grid_detection: warped img", new_img)
        cv2.waitKey(0)
    
    return new_img


def grid_detection(img, viewpoint, verbose_show=False):
    """
    It is assumed that the first pass has been done, and therefore the input image has been warped.
    
    A second preprocessing pass is performed for better lines, and then the function reconstructs
     the grid returning a data structure of the 64 squares to be passed to the neural networks.

    Parameters
    ----------
    img: numpy.ndarray
        The input chessboard image warped (oriented and transformed with the 1st pass)
    viewpoint: str {"white"|"black"}
        Viepoint of the player, the bottom part of the chessboard reserved to white/black
    verbose_show: bool
        Verbose cv.show of various step for debug/demonstration purpose

    Returns:
    --------
    squares_info: numpy.ndarray
        An array(64,4) containing information for each square:
        [
            square_counter: int [1...64]
            square coords: str ('A1', ..., 'H8')
            square_image: numpy.ndarray, img of the square
            bbox_image: numpy.ndarray, img of the piece i
        ]
    """ 
    
    assert img is not None
    bilateral_img = img.copy()
    bilateral_img = cv2.bilateralFilter(img, 3, 40, 80)
    if verbose_show:
        cv2.imshow(f"bilateral filter", bilateral_img)
        cv2.waitKey(0)

    gray = cv2.cvtColor(bilateral_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 350, apertureSize=3)
    if verbose_show:
        cv2.imshow(f"grid_detection: cannied", edges)
        cv2.waitKey(0)
    
    # Hough line prob detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 65, minLineLength=30, maxLineGap=50)
    lines = np.reshape(lines, (-1, 4))
    lines = np.array([two_points_to_polar(line) for line in lines])
    
    imgcopy = img.copy()
    if verbose_show:
        output_lines(imgcopy, lines, (0,255,0))
        cv2.imshow("grid_detection: lines prob", imgcopy)
        cv2.waitKey(0)
    
    W, H = img.shape[0] , img.shape[1]
    chessLines = ChessLines(lines, W, H)
    hLines = chessLines.getHLines()
    vLines = chessLines.getVLines()

    hLines, hLinesRemoved = removeOutLiers(hLines, grid_pass=True)
    vLines, vLinesRemoved = removeOutLiers(vLines, grid_pass=True)
    chessLines.setHLines(hLines)
    chessLines.setVLines(vLines)
    
    if verbose_show:
        imgcopy = img.copy()
        output_lines(imgcopy, hLines, (0,0,255))
        output_lines(imgcopy, vLines, (255,0,255))
        output_lines(imgcopy, hLinesRemoved, (0,0,0))
        output_lines(imgcopy, vLinesRemoved, (0,255,0))
        cv2.imshow("grid_detection: eliminazione linee outliers", imgcopy)
        cv2.waitKey(0)
    
    # clustering linee
    chessLines.cluster('DBSCAN')
    hLinesCLustered = chessLines.getHLinesClustered()
    vLinesCLustered = chessLines.getVLinesClustered()

    # linee clustered
    if verbose_show:
        imgcopy = img.copy()
        output_lines(imgcopy, hLinesCLustered , (255,0,0))
        output_lines(imgcopy, vLinesCLustered, (0,255,0))
        if len(hLinesCLustered) < 9 or len(vLinesCLustered) < 9:
            cv2.putText(imgcopy,
                        "Not enough line: DISCARDED",
                        (100,100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,0,255), 2)
            cv2.putText(imgcopy, 
                        "Provide a better image",
                        (100,150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,0,255), 2)
        for line in hLinesCLustered:
            cv2.putText(imgcopy,
                        f"{line[3]}",
                        (300,int(line[3])-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255,0,0), 2)
        for line in vLinesCLustered:
            cv2.putText(imgcopy,
                        f"{int(line[2])}",
                        (int(line[2]), 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0,0,255), 2)
        cv2.imshow("grid_detection: clustered lines", imgcopy)
        cv2.waitKey(0)
    
    
    
    # eliminazione bordi e aggiunta linee mancanti
    hLinesFinal, vLinesFinal = line_control(img, hLinesCLustered, vLinesCLustered, verbose=False)

    # abort if lines are less than expected
    if len(hLinesFinal) < 9 or len(vLinesFinal) < 9:
        print("Not enough lines found, provide a better image")
        return None
    
    if len(hLinesFinal) > 9 or len(vLinesFinal) > 9:
        print("Too much lines found, provide a better image")
        return None

    # Trova i punti in ordine
    hLinesFinal = sortLinesByDim(hLinesFinal, 3)
    vLinesFinal = sortLinesByDim(vLinesFinal, 2)
    points = intersections(hLinesFinal[:,0:2], vLinesFinal[:,0:2])
    
    if verbose_show:
        imgcopy = img.copy()
        for i, point in enumerate(points):
            x, y = (point[0], point[1])
            cv2.circle(imgcopy, (int(x),int(y)), radius=2, color=(255, 0, 0), thickness=2)
            cv2.putText(imgcopy,
                        f"{i}",
                        (int(x)+4,int(y)-4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,0,255), 2)
        cv2.imshow("points", imgcopy)
        cv2.waitKey(0)


    # Squares Extraction
    squares = extract_squares(img, points, viewpoint, debug_mode=verbose_show)
 
    if verbose_show:
        for square_coord, squareimg, bboximg in squares[:, 1:]:
            squareimg2 = squareimg.copy()
            cv2.putText(squareimg2,
                        square_coord,
                        (0, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,0,255), 1)
            cv2.imshow(f'Square', squareimg2)
            cv2.waitKey(0)

            bboximg2 = bboximg.copy()
            cv2.putText(squareimg2,
                        square_coord,
                        (0, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,0,255), 1)
            cv2.imshow(f'Square', bboximg2)
            cv2.waitKey(0)
    
    return squares


import glob, os

def main():

    error_board = []
    error_grid = []
    error = []
    analyzed_img = 0

    tommaso = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskRCNN_board() 
    model.model.load_state_dict(torch.load('./maskRCNN_epoch_2_.pth', map_location = device))
    model.to(device)
    model.eval()
    

    input_imgs = glob.glob('./input/**.png')
    print(f"INPUT IMGS : {input_imgs}")
    for input_img in input_imgs[:10]:
        analyzed_img = analyzed_img +1
        if not os.path.isfile(input_img):
            continue

        print(f"file found: {input_img}")
        imgname = input_img.split('\\')[-1]
    
        warpedBoardImg = board_detection(input_img, verbose_show=tommaso, old_version=False, model = model)
        if warpedBoardImg is None:
            error_board.append(input_img)
            error.append(input_img)
            continue

        img_grid = grid_detection(warpedBoardImg, viewpoint='White', verbose_show=tommaso)
        if img_grid is None:
            error_grid.append(input_img)
            error.append(input_img)
            continue

    print(f"Error of the board detection: {error_board}")
    print(f"Error of the grid detection: {error_grid}")
    print(f"Comulative errors {error}")
    print(f"Number of comulative errors: {len(error)}")
    print(f"Number of analized input: {analyzed_img}")
    print(f"Number of board detection: {len(error_board)}")
    print(f"Number of grid detection: {len(error_grid)}")

if __name__ == "__main__":
    main()