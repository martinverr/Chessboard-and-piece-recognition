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
    if False:
        cv2.imshow("img", img)
        cv2.waitKey(0)
    assert img is not None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if False:
        cv2.imshow("grey img", gray)
        cv2.waitKey(0)

    #gray = cv2.blur(gray, (3, 3)) # TODO auto adjust the size of the blur
    #if verbose_show:
    #    cv2.imshow("blurred grey img", gray)
    #    cv2.waitKey(0)

    # Canny edge detection

    # prova di iperparametro sigma 
   #for delta_sigma in range(10):    
    #    sigma = 0.04+0.03*delta_sigma
    #    edges = auto_canny(gray, sigma=sigma)
    #    print(f"canny sigma: {sigma}")
    #    if verbose_show:
    #        cv2.imshow(f"cannied {sigma}", edges)
    #        cv2.waitKey(0)
    #sigma=0.10
    #edges = auto_canny(gray, sigma=sigma, verbose=True)
    
    edges = cv2.Canny(gray, 70, 400, apertureSize=3)
    if False:
        cv2.imshow(f"cannied {sigma}", edges)
        cv2.waitKey(0)
    #assert np.count_nonzero(edges) / float(gray.shape[0] * gray.shape[1]) < 0.015

    # Hough line std detection
    lines1 = cv2.HoughLines(edges, 1, np.pi/180, threshold=90)
    lines1 = np.reshape(lines1, (-1, 2))
    
    # Hough line prob detection
    lines2 = cv2.HoughLinesP(edges, 1, np.pi/180, 75, minLineLength=30, maxLineGap=50)
    lines2 = np.reshape(lines2, (-1, 4))
    lines2 = np.array([two_points_to_polar(line) for line in lines2])
    
    imgcopy = img.copy()
    imgcopy2 = img.copy() 
    if False:
        output_lines(imgcopy, lines1, (0,0,255))
        cv2.imshow("lines standard", imgcopy)
        cv2.waitKey(0)
    
    if True:
        output_lines(imgcopy2, lines2, (0,255,0))
        cv2.imshow("lines prob", imgcopy2)
        cv2.waitKey(0)
    
    
    W, H = img.shape[0] , img.shape[1]
    chessLines = ChessLines(lines2, W, H)
    
    
    
    
    # Horizontal and Vertical lines
    
    # metodo manuale in base a rho
    #h, v = hor_vert_lines(lines)

    # metodo kmeans su rho
    hLines = chessLines.getHLines()
    vLines = chessLines.getVLines()

    hLines, hLinesRemoved = removeOutLiers(hLines)
    vLines, vLinesRemoved = removeOutLiers(vLines)
    chessLines.setHLines(hLines)
    chessLines.setVLines(vLines)
    
    imgcopy = img.copy()
    if False:
        output_lines(imgcopy, hLines, (0,0,255))
        output_lines(imgcopy, vLines, (0,0,255))
        output_lines(imgcopy, hLinesRemoved, (0,255,0))
        output_lines(imgcopy, vLinesRemoved, (0,255,0))
        cv2.imshow("lines standard con eliminazione outliers", imgcopy)
        cv2.waitKey(0)

    # No clustering
    #hLinesCLustered = hLines
    #vLinesCLustered = vLines
    
    # clustering linee manuale
    chessLines.cluster('DBSCAN')
    hLinesCLustered = chessLines.getHLinesClustered()
    vLinesCLustered = chessLines.getVLinesClustered()
    
        
    # linee clustered
    if True:
        imgcopy = img.copy()
        output_lines(imgcopy, hLinesCLustered , (0,0,255))
        output_lines(imgcopy, vLinesCLustered, (0,255,0))
        cv2.imshow("clustered lines", imgcopy)
        cv2.waitKey(0)

    """
    # not clustered
    # create output of the hough lines found (not clustered) onto the img
    if verbose_output:
        output_lines(img, hLines , (0,0,255))
        output_lines(img, vLines, (0,255,0))
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


    if False:
        for point in points:
            cv2.circle(img, tuple(point), 25, (0,0,255), -1)
        cv2.imwrite(f'./output/points_{output_name}', img)
    
    # Perspective transform
    new_img = four_point_transform(img_for_wrap, corners, (600, 600))
    cv2.imshow("crop", new_img)
    cv2.waitKey(0)
    """



def main():
    input_imgs = glob.glob('./input/**')
    print(input_imgs)

    for input_img in input_imgs:
        if not os.path.isfile(input_img):
            continue    

        print(f"file found: {input_img}")
        imgname = input_img.split('\\')[-1]
    
        find_board(input_img, f"{'output_' + imgname}",verbose_show=True, verbose_output=False)
        #cv2.imwrite('crop.jpg', find_board('./input/1.jpg', '1'))


if __name__ == "__main__":
    main()