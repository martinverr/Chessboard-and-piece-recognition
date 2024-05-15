import numpy as np
from time import time
import cv2

from ChessLinesClustering import ChessLines
from chessboard_detection_functions import *

import warnings
warnings.filterwarnings("ignore")

def board_detection(fname : str, verbose_show=False):
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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
    warpingSectionCorners = warpingSection(chessLines)

    # Perspective transform
    new_img = four_point_transform(img_for_wrap, warpingSectionCorners, (700, 700 + 100))
    
    if verbose_show:
        cv2.imshow("warped", new_img)
        cv2.waitKey(0)

    # return warped image of chessboard + margin
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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 65, 350, apertureSize=3)
    if verbose_show:
        cv2.imshow(f"grid_detection: cannied", edges)
        cv2.waitKey(0)
    
    # Hough line prob detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 74, minLineLength=30, maxLineGap=50)
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
    hLinesFinal, vLinesFinal = line_control(img, hLinesCLustered, vLinesCLustered, verbose=verbose_show)

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
