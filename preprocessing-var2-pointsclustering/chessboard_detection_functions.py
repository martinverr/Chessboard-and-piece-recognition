import numpy as np
import scipy.spatial as spatial
import scipy.cluster as clstr
from collections import defaultdict
from functools import partial
import cv2
import operator

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

def find_corners(points, mh, W, H):
    """
    Given a list of points, returns a list containing the four corner points.
    """
    #points = [x for x in points if x[0] >= 0 and x[0] <= H and x[1] >= 0 and x[1] <= W] #remove points with x or y <0
    if np.abs(mh) > 0.1:
        #scacchiera obliqua
        if mh > 0.1:
            bottom_right, _ = max(enumerate([pt[1] for pt in points]), key=operator.itemgetter(1))
            top_left, _ = min(enumerate([pt[1] for pt in points]), key=operator.itemgetter(1))
            bottom_left, _ = min(enumerate([pt[0] for pt in points]), key=operator.itemgetter(1))
            top_right, _ = max(enumerate([pt[0] for pt in points]), key=operator.itemgetter(1))
            corners = [points[top_left], points[top_right], points[bottom_left], points[bottom_right]]
        
        else: # mh < 0.1
            bottom_right, _ = max(enumerate([pt[0] for pt in points]), key=operator.itemgetter(1))
            top_left, _ = min(enumerate([pt[0] for pt in points]), key=operator.itemgetter(1))
            bottom_left, _ = max(enumerate([pt[1] for pt in points]), key=operator.itemgetter(1))
            top_right, _ = min(enumerate([pt[1] for pt in points]), key=operator.itemgetter(1))
            corners = [points[top_left], points[top_right], points[bottom_left], points[bottom_right]]

    else: 
        #scacchiera dritta
        # Bottom-right point has the largest (x + y) value
        # Top-left has point smallest (x + y) value
        # Bottom-left point has smallest (x - y) value
        # Top-right point has largest (x - y) value

        points = [x for x in points if (x[0] >= 0 and x[1] >= 0)] #remove points with x or y <0
        bottom_right, _ = max(enumerate([pt[0] + pt[1] for pt in points]), key=operator.itemgetter(1))
        top_left, _ = min(enumerate([pt[0] + pt[1] for pt in points]), key=operator.itemgetter(1))
        bottom_left, _ = min(enumerate([pt[0] - pt[1] for pt in points]), key=operator.itemgetter(1))
        top_right, _ = max(enumerate([pt[0] - pt[1] for pt in points]), key=operator.itemgetter(1))
        corners = [points[top_left], points[top_right], points[bottom_left], points[bottom_right]]
    return corners

def four_point_transform(img, points, dim_wrap_img):
    #pts1 = np.float32(points)
    #ritaglio con 5px di "padding"
    pts1 = np.float32([[points[0][0]-5,points[0][1]-5], [points[1][0]+5,points[1][1]-5], [points[2][0]-5,points[2][1]+5], [points[3][0]+5,points[3][1]+5]])
    pts2 = np.float32([[0, 0], [dim_wrap_img[0], 0], [0, dim_wrap_img[1]], [dim_wrap_img[0], dim_wrap_img[1]]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, matrix, (dim_wrap_img[0], dim_wrap_img[1]))


def output_lines(img, lines, color):
    for line in lines:
        rho, theta = line[0], line[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 4000*(-b))
        y1 = int(y0 + 4000*(a))
        x2 = int(x0 - 4000*(-b))
        y2 = int(y0 - 4000*(a))
        cv2.line(img,(x1,y1),(x2,y2),color,2)

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