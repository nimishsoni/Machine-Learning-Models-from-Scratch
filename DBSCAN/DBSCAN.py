import numpy as np

def euclidean_distance(x1, x2):
    """Calculates the L2 distance between two vectors."""
    return np.sqrt(np.sum((x1 - x2) ** 2))

class DBSCAN:
    def __init__(self, eps=1, min_pts=5):
        """
        Initialize DBSCAN with given parameters.

        Parameters:
        - eps: The radius within which to search for neighboring points.
        - min_pts: The minimum number of points required to form a dense region.
        """
        self.eps = eps
        self.min_pts = min_pts

    def _get_neighbours(self, sample_i):
        """
        Get neighboring points (within distance of eps) for a given sample.

        Parameters:
        - sample_i: Index of the sample for which neighbors need to be found.

        Returns:
        - neighbours: Array of indices of neighboring samples.
        """
        neighbours = []
        idxs = np.arange(len(self.X))
        for i, _sample in enumerate(self.X[idxs != sample_i]):
            if euclidean_distance(self.X[sample_i], _sample) < self.eps:
                neighbours.append(i)
        return np.array(neighbours)

    def _expand_clusters(self, sample_i, neighbours):
        """
        Expand a cluster starting from a seed sample. Expansion should include all the possible core and border neighbours

        Parameters:
        - sample_i: Index of the seed sample.
        - neighbours: Array of indices of neighboring samples.

        Returns:
        - cluster: List of indices having all the points from the expanded cluster.
        """
        cluster = [sample_i] # Initiate the new cluster with sample_i, then keep expanding as we find more neighbours
        for neighbour in neighbours:
            if neighbour not in self.visited_samples:
                self.visited_samples.append(neighbour) 

                self.neighbours[neighbour] = self._get_neighbours(neighbour) #Immediate neighbours assigned to the current (neighbour) point

                if len(self.neighbours[neighbour]) >= self.min_pts: # Checking if its a core neighbour
                    expanded_cluster = self._expand_clusters(neighbour, self.neighbours[neighbour]) #If core neighbour then it is further expanded to include all points
                    cluster += expanded_cluster #expanded cluster (list of indices of points) added to the present cluster
                else:
                    cluster.append(neighbour) # If not a core neighbour, then only present neighbour appended to existing cluster

        return cluster

    def _get_cluster_labels(self):
        """
        Assign cluster labels to the samples.

        Returns:
        - labels: Array of cluster labels corresponding to each sample.
        """
        labels = np.full(shape=self.X.shape[0], fill_value=len(self.clusters))
        for cluster_i, cluster in enumerate(self.clusters):
            for sample_i in cluster:
                labels[sample_i] = cluster_i
        return labels

    def fit_predict(self, X):
        """
        Fit the DBSCAN model to the data and return cluster labels.

        Parameters:
        - X: Input data (numpy array).

        Returns:
        - cluster_labels: Array of cluster labels for each sample.
        """
        self.X = X
        self.clusters = [] # List containing 
        self.visited_samples = [] # Running List of indices of samples already visited
        n_samples = self.X.shape[0] 
        self.neighbours = {}  # Dictionary containing mapping between a sample (as key) and its neighbours (as values)

        for sample_i in range(n_samples): # Looping over all the samples once 
            if sample_i in self.visited_samples: 
                continue

            self.neighbours[sample_i] = self._get_neighbours(sample_i) # Find neighbours of sample_i within distance eps
            
            # Separate Cluster is always started from core points having neighbours > min_pts within distance of eps
            if len(self.neighbours[sample_i]) >= self.min_pts: # Checking the condition for minimum number of points required to form dense region (or cluster)
                self.visited_samples.append(sample_i)
                # Once a core point is detected it will be expanded to include all the core + border neighbours
                new_cluster = self._expand_clusters(sample_i, self.neighbours[sample_i]) # new cluster contains list of indices of all the neighbour points

                self.clusters.append(new_cluster)  # new cluster forms a new list within self.clusters

        # final cluster labels are assigned to the samples 
        cluster_labels = self._get_cluster_labels()
        
        return cluster_labels
