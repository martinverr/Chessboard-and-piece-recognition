
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
