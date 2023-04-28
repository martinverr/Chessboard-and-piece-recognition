import numpy as np
import scipy.spatial as spatial
import scipy.cluster as clstr
from time import time
from collections import defaultdict
from functools import partial
from sklearn.utils import shuffle
import cv2
from sklearn import cluster as skcluster
import os, glob

SQUARE_SIDE_LENGTH = 227

def auto_canny(image, sigma=0.33, verbose=False):
    """
    Canny edge detection with automatic thresholds.
    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    upper = int(min(255, (1.0 + sigma) * v))
    lower = upper/3
    edged = cv2.Canny(image, lower, upper)
    if verbose:
        print(f"canny upper thr:{upper}, lower thr:{lower}")
 
    # return the edged image
    return edged

def hor_vert_lines(lines):
    """
    A line is given by rho and theta. Given a list of lines, returns a list of
    horizontal lines (theta=90 deg) and a list of vertical lines (theta=0 deg).
    """
    h = []
    v = []
    for distance, angle in lines:
        if angle < np.pi / 4 or angle > np.pi - np.pi / 4:
            v.append([distance, angle])
        else:
            h.append([distance, angle])
    return h, v

def intersections(h, v):
    """
    Given lists of horizontal and vertical lines in (rho, theta) form, returns list
    of (x, y) intersection points.
    """
    points = []
    for d1, a1 in h:
        for d2, a2 in v:
            A = np.array([[np.cos(a1), np.sin(a1)], [np.cos(a2), np.sin(a2)]])
            b = np.array([d1, d2])
            point = np.linalg.solve(A, b)
            points.append(point)
    return np.array(points)

def cluster(points, max_dist=50):
    """
    Given a list of points, returns a list of cluster centers.
    """
    Y = spatial.distance.pdist(points)
    Z = clstr.hierarchy.single(Y)
    T = clstr.hierarchy.fcluster(Z, max_dist, 'distance')
    clusters = defaultdict(list)
    for i in range(len(T)):
        clusters[T[i]].append(points[i])
    clusters = clusters.values()
    clusters = map(lambda arr: (np.mean(np.array(arr)[:,0]), np.mean(np.array(arr)[:,1])), clusters)
    return clusters

def closest_point(points, loc):
    """
    Returns the list of points, sorted by distance from loc.
    """
    dists = np.array(list(map(partial(spatial.distance.euclidean(points, loc)), points)))
    return points[dists.argmin()]

def find_corners(points, img_dim):
    """
    Given a list of points, returns a list containing the four corner points.
    """
    center_point = closest_point(points, (img_dim[0] / 2, img_dim[1] / 2))
    points.remove(center_point)
    center_adjacent_point = closest_point(points, center_point)
    points.append(center_point)
    grid_dist = spatial.distance.euclidean(np.array(center_point), np.array(center_adjacent_point))
    
    img_corners = [(0, 0), (0, img_dim[1]), img_dim, (img_dim[0], 0)]
    board_corners = []
    tolerance = 0.25 # bigger = more tolerance
    for img_corner in img_corners:
        while True:
            cand_board_corner = closest_point(points, img_corner)
            points.remove(cand_board_corner)
            cand_board_corner_adjacent = closest_point(points, cand_board_corner)
            corner_grid_dist = spatial.distance.euclidean(np.array(cand_board_corner), np.array(cand_board_corner_adjacent))
            if corner_grid_dist > (1 - tolerance) * grid_dist and corner_grid_dist < (1 + tolerance) * grid_dist:
                points.append(cand_board_corner)
                board_corners.append(cand_board_corner)
                break
    return board_corners

def four_point_transform(img, points, square_length=SQUARE_SIDE_LENGTH):
    board_length = square_length * 8
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [0, board_length], [board_length, board_length], [board_length, 0]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (board_length, board_length))


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
    lines = np.reshape(lines, (-1, 2))
    
    # Compute intersection points, first step: find lines
    
    # metodo manuale in base a rho
    #h, v = hor_vert_lines(lines)

    # metodo kmeans su linee
    angleClustering = skcluster.KMeans(n_clusters=2).fit(lines[:,1].reshape(-1,1))
    h = lines[angleClustering.labels_==0]
    v = lines[angleClustering.labels_==1]
    
    # prova con cluster agglomerativo(no n_cluster necessario)
    #clusteringH = cluster.AgglomerativeClustering(distance_threshold=(lines[:,1].min(axis=1) + lines[:,1].max(axis=1))/9) \
    #    .fit(h[:,0].reshape(-1,1))
    
    distanceClusteringH = skcluster.KMeans(n_clusters=9).fit(h[:,0].reshape(-1,1))
    distanceClusteringV = skcluster.KMeans(n_clusters=9).fit(v[:,0].reshape(-1,1))

    hMeanDists = distanceClusteringH.cluster_centers_
    vMeanDists = distanceClusteringV.cluster_centers_
    hMeanAngles = angleClustering.cluster_centers_[0].repeat(9).reshape(9,1)
    vMeanAngles = angleClustering.cluster_centers_[1].repeat(9).reshape(9,1)

    hMeanLines = np.append(hMeanDists, hMeanAngles, axis=1)
    vMeanLines = np.append(vMeanDists, vMeanAngles, axis=1)

    def output_lines(img, lines, color):
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 4000*(-b))
            y1 = int(y0 + 4000*(a))
            x2 = int(x0 - 4000*(-b))
            y2 = int(y0 - 4000*(a))
            cv2.line(img,(x1,y1),(x2,y2),color,2)
    
    # create output of the hough lines found (not clustered) onto the img
    if verbose_output:
        output_lines(img, h, (0,0,255))
        output_lines(img, v, (0,255,0))
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


def split_board(img):
    """
    Given a board image, returns an array of 64 smaller images.
    """
    arr = []
    sq_len = img.shape[0] / 8
    for i in range(8):
        for j in range(8):
            arr.append(img[i * sq_len : (i + 1) * sq_len, j * sq_len : (j + 1) * sq_len])
    return arr


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