import numpy as np

def euclidean_distance(x1, x2):
    """ Calculates the l2 distance between two vectors """
  
    return np.sqrt(np.sum((x1-x2)**2))

class DBSCAN:
    def __init__(self, eps = 1, min_pts = 5):
        self.eps = eps
        self.min_pts = min_pts

    def _get_neighbours(self, sample_i):
        neighbours = []
        idxs = np.arange(len(self.X))
        for i, _sample in enumerate(self.X[idxs != sample_i]):
            if euclidean_distance(self.X[sample_i], _sample) < self.eps:
                neighbours.append(i)

        return np.array(neighbours)
    
    def _expand_clusters(self, sample_i, neighbours):
        cluster = [sample_i]
        for neighbour in neighbours:
            if neighbour not in self.visited_samples:
                self.visited_samples.append(neighbour)
            
                self.neighbors[neighbour] = self._get_neighbours(neighbour)

                if len(self.neighbours[neighbour])>self.min_pts:
                    expanded_cluster = self._expand_cluster(neighbour,self.neighbours[neighbour])
                    cluster = cluster + expanded_cluster
                else:
                    cluster = cluster.append(neighbour)
        
        return cluster
        
    def _get_cluster_labels(self):
        """ Return the samples labels as the index of the cluster in which they are
        contained """
        # Set default value to number of clusters
        # Will make sure all outliers have same cluster label
        labels = np.full(shape=self.X.shape[0], fill_value=len(self.clusters))
        for cluster_i, cluster in enumerate(self.clusters):
            for sample_i in cluster:
                labels[sample_i] = cluster_i
        return labels

    
    def predict(self, X):
        self.X = X
        self.clusters = []
        self.visited_samples = []
        n_samples = self.X.shape[0]
        self.neighbours = {} # Dictionary containing mapping between a sample and its neighbours

        for sample_i in range(n_samples):
            if sample_i in self.visited_samples:
                continue

            self.neighbours[sample_i] = self._get_neighbours(sample_i) 

            if len(self.neighbours[sample_i]) >= self.min_pts:
                self.visited_samples.append(sample_i)

                new_cluster = self._expand_clusters(sample_i, self.neighbours[sample_i])

                self.cluster.append(new_cluster)
            
        cluster_labels = self._get_cluster_labels()
        return cluster_labels



