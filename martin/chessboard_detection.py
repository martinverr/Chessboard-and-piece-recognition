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
    if verbose_show:
        cv2.imshow("img", img)
        cv2.waitKey(0)
    assert img is not None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if verbose_show:
        cv2.imshow("grey img", gray)
        cv2.waitKey(0)

    gray = cv2.blur(gray, (3, 3)) # TODO auto adjust the size of the blur
    if verbose_show:
        cv2.imshow("blurred grey img", gray)
        cv2.waitKey(0)

    # Canny edge detection

    # prova di iperparametro sigma 
    #for delta_sigma in range(10):    
    #    sigma = 0.09+0.03*delta_sigma
    #    print(f"canny sigma: {sigma}")

    sigma=0.20
    edges = auto_canny(gray, sigma=sigma)    
    if verbose_show:
        cv2.imshow(f"cannied {sigma}", edges)
        cv2.waitKey(0)
    #assert np.count_nonzero(edges) / float(gray.shape[0] * gray.shape[1]) < 0.015

    # Hough line detection
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    
    chessLines = ChessLines(lines)
    
    # Horizontal and Vertical lines
    
    # metodo manuale in base a rho
    #h, v = hor_vert_lines(lines)

    # metodo kmeans su rho
    hLines = chessLines.getHLines()
    vLines = chessLines.getVLines()

    # No clustering
    hLinesCLustered = hLines
    vLinesCLustered = vLines
    
    # clustering linee con kmeans
    #chessLines.cluster('KmeansLines')
    #hLinesCLustered = hLines
    #vLinesCLustered = vLines
    
    
    # create output of the hough lines found (not clustered) onto the img
    if verbose_output:
        output_lines(img, hLines , (0,0,255))
        output_lines(img, vLines, (0,255,0))
        output = f'./martin/output/lines_{output_name}'
        print(f"created: {output}")
        cv2.imwrite(output, img)
    
    # calcolo intersezioni
    #points = intersections(h, v)

    # Cluster intersection points
    #points = cluster(points)
    
    # Find corners
    #img_shape = np.shape(img)
    #points = find_corners(points, (img_shape[1], img_shape[0]))
    
    if False:
        for point in points:
            cv2.circle(img, tuple(point), 25, (0,0,255), -1)
        cv2.imwrite(f'./martin/output/points_{output_name}', img)
    
    # Perspective transform
    #new_img = four_point_transform(img, points)

    #return new_img



def main():
    input_imgs = glob.glob('./martin/input/*')
    print(input_imgs)

    for input_img in input_imgs:
        if not os.path.isfile(input_img):
            continue    

        print(f"file found: {input_img}")
        imgname = input_img.split('\\')[-1]
    
        find_board(input_img, f"{'output_' + imgname}", verbose_output=True)
        #cv2.imwrite('crop.jpg', find_board('./martin/input/1.jpg', '1'))


if __name__ == "__main__":
    main()