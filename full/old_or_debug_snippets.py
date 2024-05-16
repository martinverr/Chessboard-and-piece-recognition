
gray = cv2.blur(gray, (3, 3)) # TODO auto adjust the size of the blur
if verbose_show:
    cv2.imshow("blurred grey img", gray)
    cv2.waitKey(0)


# Canny old stuff
# prova di iperparametro sigma 
for delta_sigma in range(10):    
    sigma = 0.04+0.03*delta_sigma
    edges = auto_canny(gray, sigma=sigma)
    print(f"canny sigma: {sigma}")
    if verbose_show:
        cv2.imshow(f"cannied {sigma}", edges)
        cv2.waitKey(0)
sigma=0.10
edges = auto_canny(gray, sigma=sigma, verbose=True)



# Hough line std detection
lines1 = cv2.HoughLines(edges, 1, np.pi/180, threshold=90)
lines1 = np.reshape(lines1, (-1, 2))
imgcopy = img.copy()
if False:
        output_lines(imgcopy, lines1, (0,0,255))
        cv2.imshow("lines standard", imgcopy)
        cv2.waitKey(0)


# metodo manuale in base a rho
h, v = hor_vert_lines(lines)


# No clustering
hLinesCLustered = hLines
vLinesCLustered = vLines

# codice per i punti, corner, da reinserire con la seconda passata
"""
    if False:
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
    if False:
        for point in points:
            x, y = (point[0], point[1])
            cv2.circle(img, (int(x),int(y)), radius=2, color=(0, 0, 255), thickness=2)
        cv2.imshow("clustered points", img)
        cv2.waitKey(0)
    
    
    # Find corners
    corners = find_corners(points, chessLines.mh)
    if False:
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
    """


#codice per eliminazione bordi e aggiunta righe mancanti di FB
"""
def analyze_diff(diff, mean, threshold_two_tiles = 0.1):
    index_line_eliminate = np.array([])
    index_line_to_add = np.array([])
    for index, d in enumerate(diff):
        #print(index, d, diff.shape[0])
        if d < 3*(mean / 4):
            if index in [0, len(diff) - 1]:
                print("trovato un bordo. index: "+ str(index))
                index_line_eliminate = np.append(index_line_eliminate, index)
            else:
                print("punto trovato internamente - errore")
                return
        elif d > (2 * mean - 2 * mean * threshold_two_tiles):
            index_line_to_add = np.append(index_line_to_add, index)
            if index not in [0, len(diff) - 1]:
                print("due tiles insieme - condizione in mezzo alla scacchiera")
                print(index)
            else:
                print("situazione in cui ci sono due tiles insieme - ai bordi della scacchiera")
                print(index)
        elif mean + mean / 4 <= d <= 2 * mean - 2 * mean * threshold_two_tiles:
            if index in [0, len(diff) - 1]:
                index_line_to_add = np.append(index_line_to_add, index)
                index_line_eliminate = np.append(index_line_eliminate, index+1)
                print("si tratta di un tile + bordo. index: "+ str(index))
    return index_line_eliminate, index_line_to_add

import numpy as np

def process_lines(eliminate, add, line, mean, dim, axes):

    o_line = line
    if axes not in [2,3]:
        print("errore di input su quale valore fare le operazioni nell'array delle line")
        return
    
    for e in eliminate:
        if e == 0:
            o_line = np.delete(o_line, int(e), axis=0)
        if e == line.shape[0] - 2:
            o_line = np.delete(o_line, o_line.shape[0] - 1, axis=0)
        
    for a in add:
        if axes == 2:
            dim_x = line[int(a), axes] + mean
            dim_y = 0
            dim_x_two = line[int(a), axes] + mean
            dim_y_two = dim/2
        if axes == 3:
            dim_x = 0
            dim_y = line[int(a), axes]+ mean
            dim_x_two = dim/2
            dim_y_two = line[int(a), axes]+ mean
        temp_line = np.array([dim_x, dim_y, dim_x_two, dim_y_two])
        rho, theta = two_points_to_polar(temp_line)
        o_line = np.append(o_line, [[rho, theta, 0.1, 0.1]], axis=0)

    return o_line


def line_control(img, hlines, vlines, threshold_two_tiles = 0.1, threshold_tile_plus_edge = 0.1):
    #img_copy = cv2.imread(img)
    #img_copy = np.copy(img_copy)
    img_copy =img.copy()
    W,H = img_copy.shape[0],img_copy.shape[1]
    hlines = sortLinesByDim(hlines,3)
    # proiezione dell'intersezione 
    vlines = sortLinesByDim(vlines,2)
    #print(hlines, vlines)

    hdim = hlines[:,3]
    vdim = vlines[:,2]

    h_diff = np.diff(hdim)
    v_diff = np.diff(vdim)

    print(h_diff, v_diff)

    '''
    W,H = img_copy.shape[0],img_copy.shape[1]
    for i in hlines:
        x, y = (H/2,i[3])
        cv2.circle(img_copy, (int(x),int(y)), radius=2, color=(255, 0, 0), thickness=2)
        cv2.imshow("points", img_copy)
        cv2.waitKey(0)

    for i in vlines:
        x, y = (i[2],W/2)
        cv2.circle(img_copy, (int(x),int(y)), radius=2, color=(255, 0, 0), thickness=2)
        cv2.imshow("points", img_copy)
        cv2.waitKey(0)
    '''
    
    h_mean = np.mean(h_diff)
    v_mean = np.mean(v_diff)
    #print(h_mean, v_mean)
    
    print("------------------- horizontal analysis --------------")
    h_eliminate, h_add = analyze_diff(h_diff, h_mean)
    print("------------------ vertical analysis -----------------")
    v_eliminate, v_add = analyze_diff(v_diff,v_mean)

    #print(h_eliminate, h_add, v_eliminate, v_add)

    hlines = process_lines(eliminate= h_eliminate, add=h_add, line=hlines, mean=h_mean, dim=H, axes=3)
    output_lines(img_copy, hlines, (0,0,255))
    cv2.imshow("prova", img_copy)
    cv2.waitKey(0)

    vlines = process_lines(eliminate= v_eliminate, add=v_add, line=vlines, mean=v_mean, dim=W, axes=2)
    output_lines(img_copy, vlines, (0,0,255))
    cv2.imshow("prova", img_copy)
    cv2.waitKey(0)
"""

""" grid pass lele
#mean_m = m_sum / lines.shape[0]
#mean_m = most_frequent_in_bined_array(angles_degree[1:].reshape(-1).tolist(), bin_size=1)
trimmed_count = int(len(angles_degree[1:]) * 0.1)
sorted_angles = np.sort(angles_degree[1:])
mean_m = np.mean(sorted_angles[1+trimmed_count:-trimmed_count])
"""