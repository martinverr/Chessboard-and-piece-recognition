import numpy as np
import scipy.spatial as spatial
import scipy.cluster as clstr
from collections import defaultdict
from functools import partial
import cv2
import operator
from collections import Counter


SQUARE_SIDE_LENGTH = 227


def auto_canny(image, sigma=0.33, verbose=False):
    """
    Canny edge detection with automatic thresholds.
    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    if verbose:
        print("median:", v)
    # apply automatic Canny edge detection using the computed median
    upper = int(min(255, (1.0 + sigma) * v))
    lower = int(max(0, (1.0 - 2*sigma) * v))
    
    edged = cv2.Canny(image, lower, upper, 3)
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

def find_corners(points, mh):
    """
    Given a list of points, returns a list containing the four corner points.
    """
    points = [x for x in points if (x[0] >= 0 and x[1] >= 0)] #remove points with x or y <0
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


def removeOutLiers(lines):
    angles_degree = np.ndarray(shape=(1,1), dtype=np.double)
    
    for line in lines:
        rho, theta = line[:2]
        # P0 punto proiezione da origine a retta
        x0 = np.cos(theta)*rho
        y0 = np.sin(theta)*rho
        
        # P1 punto casuale calcolato a partire da P0
        x1 = int(x0 + 1000 * (-np.sin(theta)))
        y1 = int(y0 + 1000 * (np.cos(theta)))
        
        # P2 punto casuale calcolato a partire da P0
        x2 = int(x0 - 1000 * (-np.sin(theta)))
        y2 = int(y0 - 1000 * (np.cos(theta)))
        
        """ y = mx + c """
        #TODO find a better solution for division by zero
        if x2-x1 != 0:
            m = float(y2 - y1) / (x2 - x1)
        else:
            m = 20000
        
        angle_degree = np.abs(np.degrees(np.arctan(m)))
        angles_degree = np.append(angles_degree, np.ndarray(buffer=np.array(angle_degree, dtype=np.double), shape=(1,1)), axis=0)
    
    
    #mean_m = m_sum / lines.shape[0]
    mean_m = np.mean(angles_degree[1:])
    std = np.std(angles_degree[1:])
    
    removed_lines = lines[(np.abs(angles_degree[1:] - mean_m) > 2.5 * std).reshape(-1)]
    filtered_lines = lines[(~(np.abs(angles_degree[1:] - mean_m) > 2.5 * std).reshape(-1))]
    return filtered_lines, removed_lines

def abc_line_eq_coeffs(line):
    x1, y1, x2, y2 = line

    direction_vector = np.array([x2 - x1, y2 - y1])
    normal_vector = np.array([-direction_vector[1], direction_vector[0]])
    
    # Calculate the coefficients [a, b, c] of the line equation ax+by+c
    a = normal_vector[0]
    b = normal_vector[1]
    c = -a * x1 - b * y1
    return a,b,c


def projection_point_from_origin(line):
    a, b, c = abc_line_eq_coeffs(line)
    
    length_squared = a**2 + b**2
    # Calculate the distance d from the origin to the line
    d = c / np.sqrt(length_squared)
    
    # Calculate the coordinates of the projection point
    proj_x = -a * d / np.sqrt(length_squared)
    proj_y = -b * d / np.sqrt(length_squared)

    return proj_x, proj_y

def two_points_to_polar(line):
    # Get points from the vector
    x1, y1, x2, y2 = line

    # If the line is vertical, set theta to 0 and rho to the x-coordinate of the vertical line
    if x1 == x2:
        rho = x1
        theta = 0
    else:
        proj_x, proj_y = projection_point_from_origin(line)

        # Calculate the polar coordinates of the projection point
        rho = np.sqrt(proj_x**2 + proj_y**2)
        theta = np.arctan2(proj_y, proj_x)

    #print(f"points: {(x1, y1)} and {(x2,y2)}")
    #print(f"projection equation: {a:.1f}x + {b:.1f}y + {c:.1f}")
    #print(f"projection point: {(proj_x, proj_y)}")
    #print(f"rho: {rho:.2f}, theta: {np.degrees(theta)}")
    return np.array([rho, theta])

def sortLinesByDim(lines, dim):
    return lines[lines[:, dim].argsort()]

def warpingSection(chessLines):
    """
    Find the delimiter lines of the chessboard.
    To avoid cut pieces after warp, shift those with certain margins (top:70, left:-50, right:50).

    Args:
        chessLines (np.ndarray): @see chessLines

    Returns:
        np.ndarray: 4 corners points
    """
    hLines = chessLines.getHLinesClustered()
    vLines = chessLines.getVLinesClustered()

    # Trovo il poligono da warpare
    delimiterLines =  np.empty((0,4), dtype=np.float64)
    delimiterLines = np.append(delimiterLines, sortLinesByDim(hLines, 3)[0].reshape((1,4)), axis=0) # top
    delimiterLines = np.append(delimiterLines, sortLinesByDim(hLines, 3)[-1].reshape((1,4)), axis=0) # bottom
    delimiterLines = np.append(delimiterLines, sortLinesByDim(vLines, 2)[0].reshape((1,4)), axis=0) # left
    delimiterLines = np.append(delimiterLines, sortLinesByDim(vLines, 2)[-1].reshape((1,4)), axis=0) # right


    # top, bottom, left, right margins:
    margins = [-70, 0, -50, 50]
    if np.abs(chessLines.mh) < 0.2:
        margins[2] = 0
        margins[3] = 0
    else: 
        if chessLines.mh < -0.2:
            margins[3] = 0
        if chessLines.mh > 0.2:
            margins[2] = 0
    
    #withMargin =  np.empty((0,4), dtype=np.float64)
    for i in range(4):
        if margins[i] != 0:
            # inplace delimiterLines
            delimiterLines[0] += [margins[i],0,0,0]
            
            # save in withMargin, delimiterLines untouched
            #newMarginLine = np.copy(delimiterLines[i] + [margins[i],0,0,0])
            #withMargin = np.append(withMargin, newMarginLine.reshape((1,4)), axis=0)
    
    warpSectionCorners = intersections(delimiterLines[:2,0:2], delimiterLines[2:,0:2])
    return warpSectionCorners

    
def analyze_diff(diff, mean, threshold_two_tiles = 0.1):
    index_line_eliminate = np.array([])
    index_line_to_add = np.array([])
    for index, d in enumerate(diff):
        if d < 3/4 * mean:
            if index in [0, len(diff) - 1]:
                index_line_eliminate = np.append(index_line_eliminate, index)
        elif d > 2*mean * (1 - threshold_two_tiles):
            index_line_to_add = np.append(index_line_to_add, index)
            if index not in [0, len(diff) - 1]:
                #due tiles insieme - condizione in mezzo alla scacchiera
                pass
            else:
                #due tiles insieme - ai bordi della scacchiera
                pass
        '''elif mean + mean / 4 <= d <= 2 * mean - 2 * mean * threshold_two_tiles: #else
            if index in [0, len(diff) - 1]:
                index_line_to_add = np.append(index_line_to_add, index)
                index_line_eliminate = np.append(index_line_eliminate, index+1)
                print("si tratta di un tile + bordo. index: "+ str(index))
        '''
    return index_line_eliminate, index_line_to_add


def process_lines(eliminate, add, lines, mean, dim, axes):
    o_lines = lines
    if axes not in [2,3]:
        print("errore di input su quale valore fare le operazioni nell'array delle line")
        return

    for e in eliminate:
        if e == 0:
            o_lines = np.delete(o_lines, int(e), axis=0)
        if e == lines.shape[0] - 2:
            o_lines = np.delete(o_lines, o_lines.shape[0] - 1, axis=0)

    for a in add:
        axis2, axis3 = lines[int(a), 2], lines[int(a), 3]
        if axes == 2:
            dim_x = lines[int(a), axes] + mean
            dim_y = 0
            dim_x_two = lines[int(a), axes] + mean
            dim_y_two = dim/2
            axis2 += mean
        if axes == 3:
            dim_x = 0
            dim_y = lines[int(a), axes]+ mean
            dim_x_two = dim/2
            dim_y_two = lines[int(a), axes]+ mean
            axis3 += mean
        temp_line = np.array([dim_x, dim_y, dim_x_two, dim_y_two])
        rho, theta = two_points_to_polar(temp_line)
        o_lines = np.append(o_lines, [[rho, theta, axis2, axis3]], axis=0)

    return o_lines

def most_frequent_in_bined_array(arr, bin_size = 3):
    # given x return 0, 3, 6,... depending on belonging interval [0-2.99], [3-5.99], ...
    round_to_bin = lambda x, bin_size: np.around((x / bin_size) * bin_size)
    
    # Approssima i numeri ai bin e conta le occorrenze
    rounded_arr = np.array([round_to_bin(x, bin_size) for x in arr])
    counter = Counter(rounded_arr)

    # Trova il bin con il conteggio più alto
    return counter.most_common(1)[0][0]


def line_control(img, hlines, vlines, threshold_two_tiles = 0.1, threshold_tile_plus_edge = 0.1, verbose = False):
    img_copy =img.copy()
    W,H = img_copy.shape[0],img_copy.shape[1]
    hlines = sortLinesByDim(hlines,3)
    vlines = sortLinesByDim(vlines,2)

    hdim = hlines[:,3]
    vdim = vlines[:,2]
    h_diff = np.diff(hdim)
    v_diff = np.diff(vdim)
    
    h_freq = most_frequent_in_bined_array(h_diff)
    v_freq = most_frequent_in_bined_array(v_diff)
    h_mean = np.mean(h_diff)
    v_mean = np.mean(v_diff)
    
    h_eliminate, h_add = analyze_diff(h_diff, h_mean)
    v_eliminate, v_add = analyze_diff(v_diff,v_mean)

    hlines = process_lines(eliminate= h_eliminate, add=h_add, lines=hlines, mean=h_mean, dim=H, axes=3)
    vlines = process_lines(eliminate= v_eliminate, add=v_add, lines=vlines, mean=v_mean, dim=W, axes=2)
    if verbose:
        output_lines(img_copy, hlines, (255,0,0))
        output_lines(img_copy, vlines, (0,255,0))
        
        cv2.imshow("grid_detection: lines after line_control", img_copy)
        cv2.waitKey(0)
    return hlines, vlines


"""
Calculate bounding box base on row and column

Notes
-----
h_correction =
    (c - 3) * sigma
v_correction = 
    (1 - r/8) * avg_lenght
    ir**exp_factor + linear_factor*ir + offset_term
"""
def calculate_bbox(polypoints, r, c):
    sigma = 4
    avg_lenght = 80
    offset_term = 10
    linear_factor = 7
    exp_factor = 2.2
    ir = 7-r

    h_correction = np.floor((c - 3.5) * sigma) 
    v_correction = np.floor(ir**exp_factor + linear_factor*ir + offset_term)

    polypoints_bbox = polypoints.copy()
    if h_correction > 0:
        polypoints_bbox[[1,3],0] = np.minimum(700, polypoints_bbox[[1,3],0] + h_correction)
    if h_correction < 0:
        polypoints_bbox[[0,2],0] = np.maximum(0, polypoints_bbox[[0,2],0] + h_correction)
    polypoints_bbox[:2,1] = np.maximum(0, polypoints_bbox[:1, 1] - v_correction)

    return polypoints_bbox


# works only if 9x9 lines are found
def extract_squares(img, points, viewpoint, debug_mode=False):
    """
    Extract individual squares from a chessboard image.

    Parameters
    ----------
    img: numpy.ndarray
        The input chessboard image.
    points: list
        List of corner points defining the chessboard.
    viewpoint: str
        The viewpoint of the chessboard ("white" or "black").

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
    squares_info = np.empty((64, 4), dtype=object)
    square_counter = 0

    for r in np.arange(8):
        for c in np.arange(8):            
            square_counter += 1
            letter = chr(ord('A') + c)
            number = 8 - r

            if viewpoint == "black":
                    letter = chr(ord('H') - (ord(letter) - ord('A')))
                    number = 8 - number + 1
                    
            #print(f"square {square_counter} ({letter}{number}) has corner: {r*9+c}, {r*9+c+1}, {(r+1)*9+c}, {(r+1)*9+c+1}")
            if len(points) > (r+1)*9+c+1:
                
                polypoints = np.array([points[r*9+c],
                                       points[r*9 +c+1], 
                                       points[(r+1)*9+c], 
                                       points[(r+1)*9+c+1]], np.int32)
                
                x, y, w, h = cv2.boundingRect(polypoints)
                square_image = img[y:y+h, x:x+w].copy()
                
                x, y, w, h = cv2.boundingRect(calculate_bbox(polypoints, r, c))
                bbox_image = img[y:y+h, x:x+w]

                if c-3.5 < 0:
                    bbox_image = cv2.flip(bbox_image, 1)

                square_info = np.array([square_counter, f"{letter}{number}", square_image, bbox_image], dtype=object)
                if debug_mode:
                    square_info = np.array([square_counter, f"{letter}{number}", polypoints, calculate_bbox(polypoints, r, c)], dtype=object)
                squares_info[square_counter - 1] = square_info

    return squares_info