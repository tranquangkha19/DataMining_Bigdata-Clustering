import numpy as np
import matplotlib.pyplot as plt
import csv
import hawks
import math
import timeit
from sklearn.decomposition import PCA

class KDNode:
    
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.left = None
        self.right = None

class KDTree:
    
    def __init__(self, no_dimentions):
        self.no_dimentions = no_dimentions

    def insert(self, root, key, val, coord=0):
        if not root:
            return KDNode(key, val)
        elif root.val[coord] < val[coord]:
            root.right = self.insert(root.right, key, val, (coord+1) % self.no_dimentions)
        else:
            root.left = self.insert(root.left, key, val, (coord+1) % self.no_dimentions)
        return root

    def range_search(self, val_down, val_up, node: KDNode, coord, no_dimentions, list_node, point, eps):
        if node is None:
            return
        if val_down[coord] <= node.val[coord]:
            self.range_search(val_down, val_up, node.left, (coord+1) % no_dimentions, no_dimentions, list_node, point, eps)
        coord_i = 0
        distance = 0.0
        while coord_i < no_dimentions and val_down[coord_i] <= node.val[coord_i] and val_up[coord_i] >= node.val[coord_i]:
            distance += (point[coord_i]-node.val[coord_i])**2
            coord_i = coord_i + 1
        if coord_i == no_dimentions and math.sqrt(distance) <= eps:
            list_node.append(node.key)
        if val_up[coord] >= node.val[coord]:
            self.range_search(val_down, val_up, node.right, (coord+1) % no_dimentions, no_dimentions, list_node, point, eps)
        
    def range_search_result(self, root, val_down, val_up, point, eps):
        list_node = []
        self.range_search(val_down, val_up, root, 0, self.no_dimentions, list_node, point, eps)
        return list_node


class MyDBSCAN():

    def __init__(self, no_dimentions, eps, min_points, min_elements):
        self.no_dimetions = no_dimentions
        self.eps = eps
        self.min_points = min_points
        self.min_elements = min_elements
        self.core = 'c'
        self.border = 'b'

    def neighbour_points(self, root, kdtree, point):
        val_down = []
        val_up = []
        for i in range(self.no_dimetions):
            val_down.append(point[i] - self.eps)
            val_up.append(point[i] + self.eps)
        points = kdtree.range_search_result(root, val_down, val_up, point, self.eps)
        return points

    def plot(self,n,data,point_labels):
        plt.figure(num=n,figsize=(10,10))
        pca = PCA(2)
        df = pca.fit_transform(np.array(data))
        df.shape
        label = np.array(point_labels)
        u_labels = np.unique(label)
        for i in u_labels:
            plt.scatter(df[label == i , 0] , df[label == i , 1])
        plt.show()

    def fit(self, dataset):
        
        kdtree = KDTree(self.no_dimetions)
        data = []
        root = None
        size_data = 0
        i = 0
        # Reading from the data file
        start = timeit.default_timer()
        with open(dataset) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                row_float = [float(item) for item in row[1:]]
                root = kdtree.insert(root, i, row_float)
                i += 1
                data.append(row_float)
            size_data = i
            print(f'Processed {size_data} lines.')
    
        t1 = timeit.default_timer()
        print("Thời gian để insert vào KDTree: ",t1-start)
        point_label = [-1] * size_data
        point_count = []

        # initilize list for core/border points
        
        core = []
        border = []

        # Find the neighbours of each individual point
        for i in range(size_data):
            point_count.append(self.neighbour_points(root, kdtree, data[i]))

        # Find all the core points, border points and outliers
        for i in range(size_data):
            if (len(point_count[i]) >= self.min_points):
                point_label[i] = self.core
                core.append(i)
            else:
                border.append(i)

        for i in border:
            for j in point_count[i]:
                if j in core:
                    point_label[i] = self.border
                    break
        
        t2 = timeit.default_timer()
        print("Thời gian để tìm neighbours của mỗi điểm: ",t2-t1)
                    
        # Assign points to a cluster

        cluster = 0

        # Use a stack to performing Breadth First search to find clusters of datasets

        for i in range(size_data):
            stack = []
            if (point_label[i] == self.core):
                cluster_points = [i]
                point_label[i] = cluster
                for x in point_count[i]:
                    if(point_label[x] == self.core):
                        stack.append(x)
                        point_label[x] = cluster
                        cluster_points.append(x)
                    elif(point_label[x] == self.border):
                        point_label[x] = cluster
                        cluster_points.append(x)
                while len(stack) != 0:
                    neighbors = point_count[stack.pop()]
                    for y in neighbors:
                        if (point_label[y] == self.core):
                            point_label[y] = cluster
                            stack.append(y)
                            cluster_points.append(y)
                        if (point_label[y] == self.border):
                            point_label[y] = cluster
                            cluster_points.append(y)
                if len(cluster_points) >= self.min_elements:
                    cluster += 1  # next cluster
                else:
                    for j in cluster_points:
                        point_label[j] = -1

                        
        t3 = timeit.default_timer()
        print("Thời gian để gom cụm các điểm: ",t3-t2)
        
        #hawks.plotting.scatter_prediction(np.array(data), point_label)
        #ari = adjusted_rand_score(labels, point_label)
        #print(f"ARI: {ari}")
        _data = []
        _label = []
        for i in range(len(point_label)):
            if point_label[i] != -1:
                _label += [point_label[i]]
                _data += [data[i]]
        
        t4 = timeit.default_timer()
        print("Thời gian để loại bỏ các điểm outlier: ",t4-t3)
        print("Tổng thời gian thực thi: ",t4-start)
        print("Tổng số core point và border point là: ", len(point_label))
        print("Tổng số outlier point là: ", size_data-len(_label))
        print("Tổng số cluster là: ",cluster)
        self.plot(1,data,point_label)
        self.plot(2,_data,_label)
        return data, point_label, cluster
        
def main():

    no_dimentions = 2
    eps = 0.035*100000
    min_points = 4
    min_elements = 30 #minimum elements in cluter to be recognized as a cluster
    dataset = 'birch1.txt'
    my_DBSCAN = MyDBSCAN(no_dimentions, eps, min_points, min_elements)
    
    data, point_labels, clusters = my_DBSCAN.fit(dataset)

if __name__ == "__main__":
    main()
