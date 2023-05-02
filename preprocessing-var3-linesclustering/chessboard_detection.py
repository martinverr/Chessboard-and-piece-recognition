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

    # Prepare a grey blurred img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))
    
    # Canny
    sigma=0.20
    edges = auto_canny(gray, sigma=sigma)

    # Hough line detection
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    chessLines = ChessLines(lines)
    
    # metodo kmeans su rho
    hLines = chessLines.getHLines()
    vLines = chessLines.getVLines()

    # No clustering
    hLinesCLustered = hLines
    vLinesCLustered = vLines
    
    # not clustered
    # create output of the hough lines found (not clustered) onto the img
    if verbose_output:
        output_lines(img, hLines , (0,0,255))
        output_lines(img, vLines, (0,255,0))
        output = f'./martin/output/lines_{output_name}'
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
    
    # Cluster intersection points
    avg_dist = np.abs(np.min(chessLines._v[:,2]) - np.max(chessLines._v[:,2]))/55
    points = list(cluster(points, max_dist=avg_dist))
    if verbose_show:
        for point in points:
            x, y = (point[0], point[1])
            cv2.circle(img, (int(x),int(y)), radius=2, color=(0, 0, 255), thickness=2)
        cv2.imshow("clustered points", img)
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
        #cv2.imwrite('crop.jpg', find_board('./martin/input/1.jpg', '1'))


if __name__ == "__main__":
    main()