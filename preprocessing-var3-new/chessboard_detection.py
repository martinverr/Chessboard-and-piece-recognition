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
    img_for_wrap = np.copy(img)
    assert img is not None

    # To gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Canny
    edges = cv2.Canny(gray, 70, 400, apertureSize=3)

    # Hough line std detection
    lines1 = cv2.HoughLines(edges, 1, np.pi/180, threshold=95)
    lines1 = np.reshape(lines1, (-1, 2))
    
    # Hough line prob detection
    lines2 = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=30, maxLineGap=50)
    lines2 = np.reshape(lines2, (-1, 4))
    lines2 = np.array([two_points_to_polar(line) for line in lines2])
    
    chessLines = ChessLines(lines1)
    
    # metodo kmeans su rho, divisione H e V lines
    hLines = chessLines.getHLines()
    vLines = chessLines.getVLines()
    
    # pulizia linee, elimino linee con angoli diversi
    hLines, hLinesRemoved = removeOutLiers(hLines)
    vLines, vLinesRemoved = removeOutLiers(vLines)

    #TODO: update linee pulite in chesslines, con dei setter
    
    # clustering linee manuale
    W, H = img.shape[0] , img.shape[1]
    chessLines.cluster('manual', img=img, W=W, H=H)
    hLinesCLustered = chessLines.getHLinesClustered()
    vLinesCLustered = chessLines.getVLinesClustered()
    
    # clustered
    # create output of the hough lines found (not clustered) onto the img
    if verbose_output:
        output_lines(img, hLinesCLustered, (0,0,255))
        output_lines(img, vLinesCLustered, (0,255,0))
        output = f'./output/lines_{output_name}'
        print(f"created: {output}")
        cv2.imwrite(output, img)
    
    # calcolo intersezioni su linee clusterizzate
    points = intersections(hLinesCLustered[:,0:2], vLinesCLustered[:,0:2])
    if verbose_show:
        for point in points:
            x, y = (point[0], point[1])
            cv2.circle(img, (int(x),int(y)), radius=2, color=(255, 0, 0), thickness=2)
        cv2.imshow("points", img)
        cv2.waitKey(0)
    
    # Find corners
    corners = find_corners(points, chessLines.mh)
    if verbose_show:
        for point in corners:
            x, y = (point[0], point[1])
            cv2.circle(img, (int(x),int(y)), radius=5, color=(0, 255, 0), thickness=2)
        cv2.line(img, np.int32(corners[0]), np.int32(corners[1]), (0, 255, 255), thickness=2)
        cv2.line(img, np.int32(corners[1]), np.int32(corners[3]), (0, 255, 255), thickness=2)
        cv2.line(img, np.int32(corners[2]), np.int32(corners[0]), (0, 255, 255), thickness=2)
        cv2.line(img, np.int32(corners[2]), np.int32(corners[3]), (0, 255, 255), thickness=2)
        cv2.imshow("corners", img)
        cv2.waitKey(0)
    
    # Perspective transform
    if verbose_show:
        new_img = four_point_transform(img_for_wrap, corners, (600, 600))
        cv2.imshow("crop", new_img)
        cv2.waitKey(0)



def main():
    input_imgs = glob.glob('./input/**')
    print(input_imgs)

    for input_img in input_imgs:
        if not os.path.isfile(input_img):
            continue    

        print(f"file found: {input_img}")
        imgname = input_img.split('\\')[-1]
    
        find_board(input_img, f"{'output_' + imgname}",verbose_show=False, verbose_output=False)
        #cv2.imwrite('crop.jpg', find_board('./input/1.jpg', '1'))


if __name__ == "__main__":
    main()