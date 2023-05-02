import numpy as np
from sklearn import cluster as skcluster
import cv2
from chessboard_detection_functions import output_lines



class ChessLines():
    """ Divides all 'lines' into 'h'(horizontal) and 'v'(vertical) lines
    
    Cluster lines if cluster_type is specified
        
    Parameters
    ----------

    lines : ndarray
        Supposed to be all the lines found in something like Hough
    
    cluster_type : string, default = None
        possible values are None, 'KmeansLines'
    """

    # horizontal, vertical, all ChessLines
    _h, _v, all, cluster_type = None, None, None, None

    def __init__(self, lines, cluster_type=None):
        
        self.lines = np.reshape(lines, (-1, 2))
        
        tocluster = np.column_stack((
            np.sin(self.lines[:,1]*2), 
            np.cos(self.lines[:,1]*2)
            ))
        
        angleClustering = skcluster.KMeans(n_clusters=2).fit(tocluster)
        self._angleClustering = angleClustering
        self._h = self.lines[angleClustering.labels_==0]
        self._v = self.lines[angleClustering.labels_==1]
        self.cluster_type = cluster_type

        if cluster_type is not None:
            self.cluster()
    

    def cluster(self, cluster_type=None, img=None, w=800, h=800):
        """ Update cluster_type of the class if given
        
        Cluster lines if cluster_type is specified
            
        Parameters
        ----------

        cluster_type : string, default = None
            possible values are:
                None : default, no cluster, every line is considered
                
                'KmeansLines' : cluster lines with k-means, centroids as rho and theta;
                care that the no. of group of lines must be 9 x 9

                'manual' : no ML algorithms involved (TODO has to be implemented) 
        """
            
        if cluster_type is not None:
            self.cluster_type = cluster_type
        if self.cluster_type == 'KmeansLines':
            self._h_clustered, self._v_clustered = self._KmeansLines()
        elif self.cluster_type == 'manual':
            self._h_clustered, self._v_clustered = self._manualClustering(img, w=w, h=h)

        
    def getHLines(self):
        return self._h
    
    def getVLines(self):
        return self._v

    def getHLinesClustered(self):
        return self._h_clustered
    
    def getVLinesClustered(self):
        return self._v_clustered
    

    def _KmeansLines(self):
        distanceClusteringH = skcluster.KMeans(n_clusters=9).fit(self._h[:,0].reshape(-1,1))
        distanceClusteringV = skcluster.KMeans(n_clusters=9).fit(self._v[:,0].reshape(-1,1))

        hMeanDists = distanceClusteringH.cluster_centers_
        vMeanDists = distanceClusteringV.cluster_centers_
        #hMeanAngles = self._angleClustering.cluster_centers_[0].repeat(9).reshape(9,1)
        #vMeanAngles = self._angleClustering.cluster_centers_[1].repeat(9).reshape(9,1)

        hMeanAngles = np.array([], dtype=np.float64)
        vMeanAngles = np.array([], dtype=np.float64)

        for c in np.sort(np.unique(distanceClusteringH.labels_)):
            h_cluster_line = self._h[distanceClusteringH.labels_ == c] #tutte le linee del cluster c
            hMeanAngles = np.append(hMeanAngles, np.average(h_cluster_line[:,1])) #calcolo theta medio del cluster c

        for c in np.sort(np.unique(distanceClusteringV.labels_)):
            v_cluster_line = self._v[distanceClusteringV.labels_ == c] #tutte le linee del cluster c
            vMeanAngles = np.append(vMeanAngles, np.average(v_cluster_line[:,1])) #calcolo theta medio del cluster c

        hClusteredLines = np.append(hMeanDists, hMeanAngles.reshape(9,1), axis=1)
        vClusteredLines = np.append(vMeanDists, vMeanAngles.reshape(9,1), axis=1)

        return hClusteredLines, vClusteredLines


    def _manualClustering(self, image=None, w=800, h=800, verbose=False):
        
        mh, self._h = self._addInterceptionsToLines(self._h, image=image, w=w, h=h, verbose=verbose)
        mv, self._v = self._addInterceptionsToLines(self._v, image=image, w=w, h=h, verbose=verbose)
        self.mh = mh
        self.mv = mv
        
        # horizontal have m of vertical, wrong, so swap
        if np.abs(mh) > np.abs(mv):
            tmp = self._v
            self._v = self._h
            self._h = tmp
            self.mh = mv
            self.mv = mh
        
        
        # TODO: sort horizontal lines with _sortLinesByIntersectionOnAxisY()
        # TODO: sort vertical lines with _sortLinesByIntersectionOnAxisX()
        
        # TODO: find the avg_step among clusters of line
        
        # TODO: cluster lines when we encounter a new line with a larger step than avg_step
        
        
        #debug show
        if verbose:
            output_lines(image, self._v, (0,255,0))
            cv2.imshow("clustered lines", image)
            cv2.waitKey(0)
        
        return self._h, self._v
            

    def _agglomerativeCLustering(self):
        # prova con cluster agglomerativo(no n_cluster necessario)
        clusteringH = skcluster.AgglomerativeClustering(
            distance_threshold=(self.lines[:,1].mean(axis=1))) \
                .fit(self._h[:,0].reshape(-1,1))

    def _sortLinesByRho(self):
        self._h = self._h[self._h[:, 0].argsort()]
        self._v = self._v[self._v[:, 0].argsort()]

    def _sortLinesByIntersectionOnAxisX(self):
        self._h = self._h[self._h[:, 3].argsort()]
        self._v = self._v[self._v[:, 3].argsort()]

    def _sortLinesByIntersectionOnAxisY(self):
        self._h = self._h[self._h[:, 2].argsort()]
        self._v = self._v[self._v[:, 2].argsort()]
    
    def _addInterceptionsToLines(self, lines, image=None, w=800/2, h=0, verbose=False):
        # in case we are debugging, for visual easyness, try to presort (sort by rho does not work perfectly)
        if verbose:
            self._sortLinesByRho()
        
        # vars of the previous line for the loop so that can be compared to the current one
        m_old = None
        c_old = None
        m_sum = 0
        prev_line = None
        prev_intersectionX = None
        intersections = np.ndarray(shape=(1,2), dtype=np.float32)
        
        for rho, theta in lines:
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
            c = (y2 - (m * x2))
            coefficients = [m, c]
            intersectionX = line_intersection(m, c, 0, w)[0]
            
            intersections = np.append(intersections, np.ndarray(buffer=np.array([intersectionX, c]), shape=(1,2)), axis=0)

            if verbose:
                print(f"\nline(rho, theta): {(rho, theta)}")
                print(f"\nline: {m}x + {c}")
                print(f"intersection x: {intersectionX}")
                print(f"coefficients: {coefficients}")
                if c_old is not None:
                    print(f"distance with previous point in x=0: {np.absolute(c_old - c)}")
                if prev_line is not None:
                    print(f"distance with previous line in y=0: { np.absolute(intersectionX - prev_intersectionX)}")

                # debug show what line is being analysed
                if(image is not None):
                    cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)
                    cv2.imshow(f"{(rho, theta)}", image)
                    cv2.waitKey(0)
            
            # remember as next previous line
            m_old = m
            m_sum += m
            c_old = c
            prev_line = [rho, theta]
            prev_intersectionX = intersectionX
        
        
        m = m_sum / lines.shape[0]
        return m, np.append(lines, intersections[1:], axis=1)
        
        #TODO return m angular coef, mean or last (shoulb be good enough)



def line_intersection(m1, c1, m2, c2):
    # Check if lines are parallel
    if m1 == m2:
        return None

    # Compute x- and y-coordinates of intersection point
    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1

    return (x, y)