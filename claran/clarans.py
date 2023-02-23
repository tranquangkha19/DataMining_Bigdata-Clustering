import math
import random


def euclidean_distance(point1, point2):
    # point1, point2 are 2 arrays
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


class CLARANS:

    def __init__(self, data, number_clusters, numlocal, maxneighbor):
        """!
        The higher the value of maxneighbor, the closer is CLARANS to K-Medoids, and the longer is each search of a local minima.

        @param[in] data: Input data that is presented as list of points
        @param[in] number_clusters: Amount of clusters that should be allocated.
        @param[in] numlocal: The number of local minima obtained (amount of iterations for solving the problem).
        @param[in] maxneighbor: The maximum number of neighbors examined.

        """

        self.data = data
        self.numlocal = numlocal
        self.maxneighbor  = maxneighbor
        self.number_clusters = number_clusters

        self.clusters = []
        self.current = []
        self.belong = []

        self.optimal_medoids = []
        self.optimal_estimation = float('inf')

    def process(self):
        random.seed()

        for _ in range(0, self.numlocal):
            # set (current) random medoids
            self.current = random.sample(range(0, len(self.data)), self.number_clusters)

            # update clusters in line with random allocated medoids
            self.update_clusters(self.current)

            # optimize configuration
            self.optimize_configuration()

            # obtain cost of current cluster configuration and compare it with the best obtained
            estimation = self.calculate_estimation()
            if estimation < self.optimal_estimation:
                self.optimal_medoids = self.current[:]
                self.optimal_estimation = estimation

        self.update_clusters(self.optimal_medoids)
        return self

    def update_clusters(self, medoids):
        """!
        Forms cluster with specified medoids by calculation distance from each point to medoids.
        """

        self.belong = [0] * len(self.data)
        self.clusters = [[] for i in range(len(medoids))]
        for idx_point in range(len(self.data)):
            idx_optim = -1
            dist_optim = 0.0

            for idx in range(len(medoids)):
                dist = euclidean_distance(self.data[idx_point], self.data[medoids[idx]])

                if (dist < dist_optim) or (idx == 0):
                    idx_optim = idx
                    dist_optim = dist

            self.clusters[idx_optim].append(idx_point)
            self.belong[idx_point] = idx_optim

        # If cluster is not able to capture object it should be removed
        self.clusters = [cluster for cluster in self.clusters if len(cluster) > 0]

    def optimize_configuration(self):
        """!
        Finds quasi-optimal medoids and updates clusters with algorithm's rules.

        """
        idx_neighbor = 0
        while idx_neighbor < self.maxneighbor:
            # get random current medoid that is to be replaced
            current_medoid_idx = self.current[random.randint(0, self.number_clusters - 1)]
            current_medoid_cluster_idx = self.belong[current_medoid_idx]

            # get new candidate to be medoid
            candidate_medoid_idx = random.randint(0, len(self.data) - 1)

            while candidate_medoid_idx in self.current:
                candidate_medoid_idx = random.randint(0, len(self.data) - 1)

            candidate_cost = 0.0
            for point_idx in range(0, len(self.data)):
                if point_idx not in self.current:
                    # get non-medoid point and its medoid
                    point_cluster_idx = self.belong[point_idx]
                    point_medoid_idx = self.current[point_cluster_idx]

                    # get other medoid that is nearest to the point (except current and candidate)
                    other_medoid_idx = self.find_another_nearest_medoid(point_idx, current_medoid_idx)
                    other_medoid_cluster_idx = self.belong[other_medoid_idx]

                    # distance from the point to current medoid
                    distance_current = euclidean_distance(self.data[point_idx],
                                                          self.data[current_medoid_idx])

                    # distance from the point to candidate median
                    distance_candidate = euclidean_distance(self.data[point_idx],
                                                            self.data[candidate_medoid_idx])

                    # distance from the point to nearest (own) medoid
                    distance_nearest = float('inf')
                    if ((point_medoid_idx != candidate_medoid_idx) and (
                            point_medoid_idx != current_medoid_cluster_idx)):
                        distance_nearest = euclidean_distance(self.data[point_idx],
                                                              self.data[point_medoid_idx])

                    # apply rules for cost calculation
                    if point_cluster_idx == current_medoid_cluster_idx:
                        # case 1:
                        if distance_candidate >= distance_nearest:
                            candidate_cost += distance_nearest - distance_current

                        # case 2:
                        else:
                            candidate_cost += distance_candidate - distance_current

                    elif point_cluster_idx == other_medoid_cluster_idx:
                        # case 3:
                        if distance_candidate > distance_nearest:
                            pass;

                        # case 4:
                        else:
                            candidate_cost += distance_candidate - distance_nearest

            if candidate_cost < 0:
                self.current[current_medoid_cluster_idx] = candidate_medoid_idx

                # recalculate clusters
                self.update_clusters(self.current)

                # reset iterations and starts investigation from the begining
                idx_neighbor = 0

            else:
                idx_neighbor += 1

    def find_another_nearest_medoid(self, point_idx, current_medoid_idx):
        """!
        Finds the another nearest medoid for the specified point that is differ from the specified medoid.
        """
        other_medoid_idx = -1
        other_distance_nearest = float('inf')
        for idx_medoid in self.current:
            if idx_medoid != current_medoid_idx:
                other_distance_candidate = euclidean_distance(self.data[point_idx],
                                                              self.data[current_medoid_idx])

                if other_distance_candidate < other_distance_nearest:
                    other_distance_nearest = other_distance_candidate
                    other_medoid_idx = idx_medoid

        return other_medoid_idx

    def calculate_estimation(self):
        """!
        Calculates estimation (cost) of the current clusters. The lower the estimation, the more optimally
        configuration of clusters.
        """
        estimation = 0.0
        for idx_cluster in range(0, len(self.clusters)):
            cluster = self.clusters[idx_cluster]
            idx_medoid = self.current[idx_cluster]
            for idx_point in cluster:
                estimation += euclidean_distance(self.data[idx_point],
                                                 self.data[idx_medoid])

        return estimation
