import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def draw_lines(img, lines, color=(0,0,0)):

    for r, theta in lines:

        # Stores the value of cos(theta) in a
        a = np.cos(theta)
    
        # Stores the value of sin(theta) in b
        b = np.sin(theta)
    
        # x0 stores the value rcos(theta)
        x0 = a*r
    
        # y0 stores the value rsin(theta)
        y0 = b*r
    
        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000*(-b))
    
        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000*(a))
    
        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000*(-b))
    
        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000*(a))
    
        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        # (0,0,255) denotes the colour of the line to be
        # drawn. In this case, it is red.
        cv2.line(img, (x1, y1), (x2, y2), color, 2)
    return img

img = cv2.imread('test_6.jpg')

img = cv2.resize(img, (800, 800))
img_clone = np.copy(img)
img_lines = np.copy(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel_size = 3
gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

edges = cv2.Canny(gray, 50, 150, apertureSize=3)

lines = cv2.HoughLines(edges, 1, np.pi/180, 150)  # r, theta
lines = lines.reshape(-1,2)
arr_r = lines[:,0]

arr_theta = np.column_stack((np.sin(lines[:,1]),np.abs(np.cos(lines[:,1]))))

kmeans_h_v = KMeans(n_clusters=2)
kmeans_h_v.fit(arr_theta)
plt.scatter(arr_theta[:,0], arr_theta[:,1], c=kmeans_h_v.labels_)
plt.show()

h_lines = lines[kmeans_h_v.labels_ == 0]
v_lines = lines[kmeans_h_v.labels_ == 1]

#img_lines = draw_lines(img_lines, h_lines, color=(0,0,255)) #disegno linee orizzontali
#img_lines = draw_lines(img_lines, v_lines, color=(0,255,0)) #disegno linee verticali

kmeans_h = KMeans(n_clusters = 9)
kmeans_h.fit(np.reshape(h_lines[:,0], (-1,1)))
plt.scatter(h_lines[:,0], np.ones(h_lines.shape[0])*10, c=kmeans_h.labels_)
plt.show()

kmeans_v = KMeans(n_clusters = 9)
kmeans_v.fit(np.reshape(v_lines[:,0], (-1,1)))
plt.scatter(v_lines[:,0], np.ones(v_lines.shape[0])*10, c=kmeans_v.labels_)
plt.show()

h_theta = np.array([], dtype=np.float64)
v_theta = np.array([], dtype=np.float64)

for c in np.sort(np.unique(kmeans_h.labels_)):
    h_cluster_line = h_lines[kmeans_h.labels_ == c] #tutte le linee del cluster c
    h_theta = np.append(h_theta, np.average(h_cluster_line[:,1])) #calcolo theta medio del cluster c

for c in np.sort(np.unique(kmeans_v.labels_)):
    v_cluster_line = v_lines[kmeans_v.labels_ == c] #tutte le linee del cluster c
    v_theta = np.append(v_theta, np.average(v_cluster_line[:,1])) #calcolo theta medio del cluster c
    
lines_tmp_h = np.column_stack((kmeans_h.cluster_centers_, np.ones(9)*h_theta))
lines_tmp_v = np.column_stack((kmeans_v.cluster_centers_, np.ones(9)*v_theta))

img_lines = draw_lines(img_lines, lines_tmp_h, color=(0,0,255))
img_lines = draw_lines(img_lines, lines_tmp_v, color=(0,255,0))

cv2.imshow('linesDetected', img_lines)
cv2.waitKey(0)