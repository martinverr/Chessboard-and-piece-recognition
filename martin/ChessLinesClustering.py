import numpy as np
from sklearn import cluster as skcluster


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
        
        angleClustering = skcluster.KMeans(n_clusters=2).fit(self.lines[:,1].reshape(-1,1))
        self._angleClustering = angleClustering
        self._h = self.lines[angleClustering.labels_==0]
        self._v = self.lines[angleClustering.labels_==1]
        self.cluster_type = cluster_type

        if cluster_type is not None:
            self.cluster()
    

    def cluster(self, cluster_type=None):
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
            self._h_clustered, self._v_clustered = self._manualClustering()

        
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

    def _manualCLustering(self):
        pass
    

    def _agglomerativeCLustering(self):
        # prova con cluster agglomerativo(no n_cluster necessario)
        clusteringH = skcluster.AgglomerativeClustering(
            distance_threshold=(self.lines[:,1].mean(axis=1))) \
                .fit(self._h[:,0].reshape(-1,1))