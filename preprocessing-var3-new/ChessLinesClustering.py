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
    

    def cluster(self, cluster_type=None, img=None, W=800, H=800):
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
            self._h_clustered, self._v_clustered = self._manualClustering(img, W=W, H=H)

        
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


    def _manualClustering(self, image=None, W=800, H=800, verbose=False):
        
        mh, self._h = self._addInterceptionsToLines(self._h, image=image, W=W, H=H, verbose=verbose)
        mv, self._v = self._addInterceptionsToLines(self._v, image=image, W=W, H=H, verbose=verbose)
        self.mh = mh
        self.mv = mv
        
        # horizontal have m of vertical, wrong, so swap
        if np.abs(mh) > np.abs(mv):
            tmp = self._v
            self._v = self._h
            self._h = tmp
            self.mh = mv
            self.mv = mh
        
        self._sortHLinesByIntersectionY()
        self._sortVLinesByIntersectionX() 

        v_steps = np.abs(self._v[1:, 2] - self._v[:-1, 2])
        v_avg_step = np.sum(v_steps) / self._v.shape[0]
        
        h_steps = np.abs(self._h[1:, 3] - self._h[:-1, 3])
        h_avg_step = np.sum(h_steps) / self._h.shape[0]
        
        if verbose:
            output_lines(image, self._v, [0,0,255])
            print("\nIntersezioni linee verticali:")
            print(self._v[:,2])
            print("Steps:")
            print(v_steps)
            print(v_avg_step)
        
        if verbose:
            output_lines(image, self._h, [0,255,0])
            print("\nIntersezioni linee orizzontali:")
            print(self._h[:,3])
            print("Steps:")
            print(h_steps)
            print(h_avg_step)
        
        return self._manualClustering_on(self._h, h_avg_step, 3, tolerance=0.7), \
                self._manualClustering_on(self._v, v_avg_step, 2, tolerance=0.7)
            

    def _agglomerativeCLustering(self):
        # prova con cluster agglomerativo(no n_cluster necessario)
        clusteringH = skcluster.AgglomerativeClustering(
            distance_threshold=(self.lines[:,1].mean(axis=1))) \
                .fit(self._h[:,0].reshape(-1,1))

    def _sortLinesByRho(self):
        self._h = self._h[self._h[:, 0].argsort()]
        self._v = self._v[self._v[:, 0].argsort()]

    def _sortHLinesByIntersectionY(self):
        self._h = self._h[self._h[:, 3].argsort()]

    def _sortVLinesByIntersectionX(self):
        self._v = self._v[self._v[:, 2].argsort()]
    
    def _addInterceptionsToLines(self, lines, image=None, W=800, H=800, verbose=False):
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
            intersectionX = line_intersection(m, c, 0, H/2)[0]
            intersectionY = m * W/2 + c
            
            intersections = np.append(intersections, np.ndarray(buffer=np.array([intersectionX, intersectionY]), shape=(1,2)), axis=0)

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
        
    def _manualClustering_on(self, lines, avg_step, cmpdim, tolerance=0.5, verbose=False, image=None):
        dividing_step = avg_step * tolerance
        clustered_lines = np.zeros(shape=(1,4), dtype=np.float64)
        #np.append(intersections, np.ndarray(buffer=np.array([intersectionX, c]), shape=(1,2)), axis=0)
        buffer_cluster_lines = np.zeros(shape=(1,4), dtype=np.float64)
        clusterLine_counter = 0
        for counter, currLine in enumerate(lines):
            if counter == 0:
                buffer_cluster_lines[0] = currLine
                clusterLine_counter = 1
                continue
            
            if currLine[cmpdim] - lines[counter-1][cmpdim] < dividing_step:
                if clusterLine_counter == 0:
                    buffer_cluster_lines[0] = currLine.reshape(1,4)
                    clusterLine_counter = 1
                else:
                    buffer_cluster_lines = np.append(buffer_cluster_lines, currLine.reshape(1,4), axis=0)
                    clusterLine_counter += 1
            else:
                if clusterLine_counter == 0:
                    clustered_line = currLine.reshape(1,4)
                    clustered_lines = np.append(clustered_lines, clustered_line, axis = 0)
                    clusterLine_counter = 0
                else:
                    clustered_line = np.mean(buffer_cluster_lines, axis=0, dtype=np.float64).reshape(1,4)
                    clustered_lines = np.append(clustered_lines, clustered_line, axis = 0)
                    buffer_cluster_lines = currLine.reshape(1,4)
                    clusterLine_counter = 1
                #debug show
                if verbose:
                    print(buffer_cluster_lines)
                    print("sono state clusterizzate in:")
                    print(clustered_line)
        
        if clusterLine_counter == 0:
            clustered_line = currLine.reshape(1,4)
        else:
            clustered_line = np.mean(buffer_cluster_lines, axis=0, dtype=np.float64).reshape(1,4)
        clustered_lines = np.append(clustered_lines, clustered_line, axis = 0)
        clustered_lines = clustered_lines[1:]
        if verbose:
            output_lines(image, clustered_line, (0,255,0))
            cv2.imshow(f"clustered lines manual{counter}", image)
            cv2.waitKey(0)
        
        return clustered_lines



def line_intersection(m1, c1, m2, c2):
    # Check if lines are parallel
    if m1 == m2:
        raise Exception("intersection between parallel line")

    # Compute x- and y-coordinates of intersection point
    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1

    return (x, y)