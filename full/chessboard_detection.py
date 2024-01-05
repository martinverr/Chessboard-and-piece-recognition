import numpy as np
from time import time
import cv2
import os, glob

from ChessLinesClustering import ChessLines
from chessboard_detection_functions import *
from FEN import FEN

import warnings
warnings.filterwarnings("ignore")

def board_detection(fname, output_name, verbose_show=False, verbose_output=False):
    """
    Given a filename, returns the board image.
    """
    start = time()
    img = cv2.imread(fname)
    img_for_wrap = np.copy(img)
    if verbose_show:
        cv2.imshow("img", img)
        cv2.waitKey(0)
    assert img is not None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if verbose_show:
        cv2.imshow("grey img", gray)
        cv2.waitKey(0)

    edges = cv2.Canny(gray, 70, 400, apertureSize=3)
    if verbose_show:
        cv2.imshow(f"cannied", edges)
        cv2.waitKey(0)
    #assert np.count_nonzero(edges) / float(gray.shape[0] * gray.shape[1]) < 0.015

    # Hough line prob detection
    lines2 = cv2.HoughLinesP(edges, 1, np.pi/180, 69, minLineLength=30, maxLineGap=50)
    lines2 = np.reshape(lines2, (-1, 4))
    lines2 = np.array([two_points_to_polar(line) for line in lines2])
    
    imgcopy2 = img.copy()     
    if verbose_show:
        output_lines(imgcopy2, lines2, (0,255,0))
        cv2.imshow("lines prob", imgcopy2)
        cv2.waitKey(0)
    
    
    W, H = img.shape[0] , img.shape[1]
    chessLines = ChessLines(lines2, W, H)
    
    # Horizontal and Vertical lines
    # metodo kmeans su rho
    hLines = chessLines.getHLines()
    vLines = chessLines.getVLines()

    # Pulizia linee
    # TODO: forse il parametro dei removeOutLiers con il nuovo dataset e' da rivedere, a primo impatto non mi sembra
    hLines, hLinesRemoved = removeOutLiers(hLines)
    vLines, vLinesRemoved = removeOutLiers(vLines)
    chessLines.setHLines(hLines)
    chessLines.setVLines(vLines)
    
    if verbose_show:
        imgcopy = img.copy()
        output_lines(imgcopy, hLines, (0,0,255))
        output_lines(imgcopy, vLines, (0,0,255))
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
                        cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0,0,255), 3)
            cv2.putText(imgcopy, 
                        "Provide a better image",
                        (100,200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0,0,255), 3)
        cv2.imshow("clustered lines", imgcopy)
        cv2.waitKey(0)
            
    
    # abort if clustered lines are less than expected
    if len(hLinesCLustered) < 9 or len(vLinesCLustered) < 9:
            print("Not enough lines found, provide a better image")
            return None
    
    # 4 corner della sezione dell'immagine da warpare
    warpingSectionCorners = warpingSection(chessLines)

    # Perspective transform
    #TODO: +100 forse da modificare (calcolare con una proporzione ed euristica?) 
    new_img = four_point_transform(img_for_wrap, warpingSectionCorners, (700, 700 + 100))
    
    if verbose_show:
        cv2.imshow("warped", new_img)
        cv2.waitKey(0)

    
    # return warped image of chessboard + margin
    return new_img


def grid_detection(img, viewpoint, verbose_show=False):
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

    hLines, hLinesRemoved = removeOutLiers(hLines)
    vLines, vLinesRemoved = removeOutLiers(vLines)
    chessLines.setHLines(hLines)
    chessLines.setVLines(vLines)
    """
    if verbose_show:
        imgcopy = img.copy()
        output_lines(imgcopy, hLines, (0,0,255))
        output_lines(imgcopy, vLines, (0,0,255))
        output_lines(imgcopy, hLinesRemoved, (0,255,0))
        output_lines(imgcopy, vLinesRemoved, (0,255,0))
        cv2.imshow("grid_detection: eliminazione linee outliers", imgcopy)
        cv2.waitKey(0)
    """
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
    if len(hLinesFinal) < 9 or len(vLines) < 9:
            print("Not enough lines found, provide a better image")
            return None
        
    # Trova i punti in ordine
    hLinesFinal = sortLinesByDim(hLinesFinal, 3)
    vLinesFinal = sortLinesByDim(vLinesFinal, 2)
    points = intersections(hLinesFinal[:,0:2], vLinesFinal[:,0:2])
    
    if True:
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
    squares = extract_squares(img, points, viewpoint)
    if True:
        for squarename, squareimg in squares.items():
            cv2.namedWindow(f'Square', cv2.WINDOW_NORMAL)  # Use WINDOW_NORMAL to allow resizing
            cv2.resizeWindow(f'Square', 200, 200)
            squareimg2 = squareimg.copy()
            cv2.putText(squareimg2,
                        squarename,
                        (0, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,0,255), 1)
            cv2.imshow(f'Square', squareimg2)
            cv2.waitKey(0)
    
    return squares
                

                


def classify():
    return {}


def main():
    #195
    input_imgs = glob.glob('./input/**')
    print(input_imgs)

    for input_img in input_imgs:
        if not os.path.isfile(input_img):
            continue
        
        if not input_img.lower().endswith(".png"):
            continue
        
        print(f"file found: {input_img}")
        imgname = input_img.split('\\')[-1]
    
        warpedBoardImg = board_detection(input_img, 
                                         f"{'output_' + imgname}",
                                         verbose_show=False, 
                                         verbose_output=False)
        if warpedBoardImg is None:
            continue


        predicted_pos = classify()
        truth = FEN(os.path.splitext(input_img)[0])
        true_fen, true_pos, viewpoint = truth.fen, truth.pieces, truth.view
        
        grid_detection(warpedBoardImg,
                       viewpoint,
                       verbose_show=False)
        
        #debug lettura json
        if False:
            print("\n##### DEBUG JSON START ######\n")
            print(f"FEN: {true_fen}\n\nPieces Position: {true_pos}\n\nViewpoint: {viewpoint}")
            print("\n##### DEBUG JSON END ######\n")


if __name__ == "__main__":
    main()