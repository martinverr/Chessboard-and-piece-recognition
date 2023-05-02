import numpy as np
from time import time
import cv2
import os, glob

from ChessLinesClustering import ChessLines
from chessboard_detection_functions import *


def find_board(fname, output_name, verbose_show=False, verbose_output=False):
    """
    Given a filename, returns the board image.
    """
    start = time()
    img = cv2.imread(fname)
    assert img is not None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if verbose_show:
        cv2.imshow("blurred grey img", gray)
        cv2.waitKey(0)

    # Canny edge detection
    sigma=0.20
    edges = auto_canny(gray, sigma=sigma)    
    if verbose_show:
        cv2.imshow(f"cannied {sigma}", edges)
        cv2.waitKey(0)
    
    # Apply Laplace function
    edges2 = cv2.Laplacian(gray, ddepth = cv2.CV_16S, ksize=3)
    edges2 = cv2.convertScaleAbs(edges2)
    if verbose_show:
        cv2.imshow("laplacian", edges2)
        cv2.waitKey(0)

    # Hough line detection
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    chessLines = ChessLines(lines)
    
    # Hough line detection
    lines2 = cv2.HoughLines(edges2, 1, np.pi/180, 100)
    chessLines2 = ChessLines(lines2)

    imgcopy = img.copy()
    imgcopy2 = img.copy()
    if verbose_show:
        output_lines(imgcopy, chessLines.lines, (0,0,255))
        output_lines(imgcopy2, chessLines2.lines, (0,0,255))
        cv2.imshow("lines", imgcopy)
        cv2.waitKey(0)
        cv2.imshow("lines with laplacian", imgcopy2)
        cv2.waitKey(0)
    



def main():
    input_imgs = glob.glob('./martin/input/**')
    print(input_imgs)

    for input_img in input_imgs:
        if not os.path.isfile(input_img):
            continue    

        print(f"file found: {input_img}")
        imgname = input_img.split('\\')[-1]
    
        find_board(input_img, f"{'output_' + imgname}",verbose_show=True, verbose_output=False)

if __name__ == "__main__":
    main()