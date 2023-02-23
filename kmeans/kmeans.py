import numpy as np

class KMeansPlusPlus:
    def __init__(self, n_clusters=5, verbose=True):
        self.n_clusters = n_clusters
        self.verbose = verbose
    
    def init_centers(self, X):
        centers = []
        centers.append(X[np.random.randint(X.shape[0])])
        
        for i in range(self.n_clusters - 1):
            differences = X[:, None] - centers
            min_distances = np.min(np.linalg.norm(differences, axis=2), axis=1)
            squared_distances = np.power(min_distances, 4)
            probabilities = squared_distances / np.sum(squared_distances)
            new_center_index = np.random.choice(X.shape[0], None, False, probabilities)
            centers.append(X[new_center_index])
        return np.array(centers)

            

class KMeans:
    def __init__(self, n_clusters=5, max_iter=1000, verbose=True, init='random', tol=0.0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.verbose = verbose
        self.init = init
        self.tol = tol
        
    def _init_centers(self, X):
        if self.init == 'kmeansplusplus':
            kminit = KMeansPlusPlus(self.n_clusters)
            self.cluster_centers_ = kminit.init_centers(X)
        elif self.init == 'random':
            self.cluster_centers_ = X[np.random.choice(X.shape[0], self.n_clusters, False)]
        else:
            raise Exception("init method not correct")
        
    def _assign_labels(self, X):
        self.label_ = []
        self.error_mean_ = 0
        for point in X:
            distances = np.linalg.norm(self.cluster_centers_ - point, axis = 1)
            min_distance_idx =np.argmin(distances)
            self.label_.append(min_distance_idx)
        self.label_ = np.array(self.label_)
            
            
    def _calculate_centers(self, X):
        self._old_centers = self.cluster_centers_
        
        sum_points = np.zeros(self.cluster_centers_.shape)
        num_points = np.zeros((self.cluster_centers_.shape[0], 1))
        for idx, point in enumerate(X):
            point_label = self.label_[idx]
            sum_points[point_label] = sum_points[point_label] + point
            num_points[point_label][0] = num_points[point_label][0] + 1
            
        self.cluster_centers_ = sum_points / num_points
        
        
        
    def _inerita(self, X):
        return np.sum(np.power(X - self.cluster_centers_[self.label_], 2))
    
    def fit(self, X, verbose = None):
        if verbose is None:
            verbose = self.verbose
            
        self._init_centers(X)
        self._old_centers = np.zeros(self.cluster_centers_.shape)
        self._assign_labels(X)
        for i in range(self.max_iter):
            last_labels = self.label_
            self._calculate_centers(X)
            self._assign_labels(X)
            
            labels_changed = np.sum(last_labels != self.label_)
            
            if (verbose):
                print("Iteration ", i, "/", self.max_iter)
                print("Labels_changed : ", labels_changed)
            
            if (self.tol != 0.0 and np.sum(np.power(self._old_centers - self.cluster_centers_, 2)) < self.tol):
                print("stopped because of tol")
                print("number of iter : ", i)
                break
            
            if labels_changed == 0:
                print("number of iter : ", i)
                break
        else:
            print("number of iter : ", self.max_iter)
        
        self.inerita_ = self._inerita(X)
    
    def predict(self, X):
        print(X)
        
class MiniBatchKMeans(KMeans):
    def __init__(self, n_clusters=5, max_iter=1000, verbose=True, init='random', tol=0.0, batch_size=1024, stop_size=-1):
        super().__init__(n_clusters, max_iter, verbose, init, tol)
        self.batch_size = batch_size
        self.stop_size = stop_size
        
    def fit(self, X, verbose = None):
        if verbose is None:
            verbose = self.verbose
            
        self._init_centers(X)
        self._old_centers = np.zeros(self.cluster_centers_.shape)
        v = np.zeros(self.cluster_centers_.shape[0])
        self.label_ = np.zeros((X.shape[0],))
        
        for i in range(self.max_iter):
            if (verbose):
                print("Iteration: ", i, "/", self.max_iter)
            batch_indexs = np.random.choice(X.shape[0], self.batch_size, False)
            label_changed = 0
            
            for idx in batch_indexs:
                distances = np.linalg.norm(self.cluster_centers_ - X[idx], axis = 1)
                min_distances_idx = np.argmin(distances)
                if self.label_[idx] != min_distances_idx:
                    label_changed = label_changed + 1
                    self.label_[idx] = min_distances_idx
                    
            if (verbose):
                print("Label changed: ", label_changed)
                    
            if (self.tol != 0.0 and np.sum(np.power(self.cluster_centers_ - self._old_centers, 2)) < self.tol):
                print("Stopped because of tol")
                print("Sum: ", np.sum(np.power(self.cluster_centers_ - self._old_centers, 2)))
                print("Num of inter: ", i)
                break
            elif (self.tol == 0.0):
                if label_changed == 0 or label_changed < self.stop_size:
                    print("Stopped because no label changed or lower than stop_size")
                    print("Num of inter: ", i)
                    break
                    
            self._old_centers = np.copy(self.cluster_centers_)
            
            for idx in batch_indexs:
                center_idx = int(self.label_[idx])
                v[center_idx] = v[center_idx] + 1
                lr = 1 / v[center_idx]
                self.cluster_centers_[center_idx] = (1 - lr) * self.cluster_centers_[center_idx] + lr*X[idx]
        
        for idx, point in enumerate(X):
            distances = np.linalg.norm(self.cluster_centers_ - X[idx], axis = 1)
            min_distances_idx = np.argmin(distances)
            self.label_[idx] = min_distances_idx
        
        self.label_ = self.label_.astype(int)
        self.inerita_ = self._inerita(X)
            
    
    def predict(self, X):
        pass
    