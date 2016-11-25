# create by zzy

import numpy as np
from tools import get_dis, equals

class kmeans():
    # data : np.mat.tolist()[0]
    # K : K means
    def __init__(self, data, K):
        self.data = data
        self.x = len(data)
        self.y = len(data[0])
        self.K = K
        self.labels = [-1 for i in range(self.X)]
        self.centroids = self.__initCentroids()

    def __initCentroids(self):
        lst = []
        for i in range(self.K):
            lst.append(self.data[i])
        return lst

    def _kmeans(self):
        flag = True
        while True:
            flag = False
            old_centroids = self.centroids
            self._E_step()
            self._M_step()
            for i in xrange(len(old_centroids)):
                if not equals(old_centroids[i], self.centroids[i]):
                    flag = True


    # tag labels
    def _E_step(self):
        for i in xrange(len(self.data)):
            min = get_dis(self.data[i], self.centroids[0])
            for j in xrange(self.K):
                tmp = get_dis(self.data[i], self.centroids[j])
                if tmp <= min :
                    min = tmp
                    self.label[i] = j


    # caculate centroids
    def _M_step(self):
        lst = []
        for j in xrange(self.K):
            lst.append([self.data[i] for i in xrange(self.x) and self.labels[i] == j])
        for j in xrange(self.K):
            self.centroids[j] = self._caculate_centroids(lst[j])

    def _caculate_centroids(self, points):
        cen = points[0]
        len = len(points)
        for i in xrange(len):
            x = points[i]
            sum = 0
            for j in range(i, len):
                y = points[j]
                sum += get_dis(x,y)
            if i == 0:
                min = sum
            if sum <= min:
                min = sum
                cen = points[i]
        return cen


    def get_arg(self):

        pass

if __name__ == '__main__':
    kmeans = kmeans([[1,1],[1,2],[4,5]],2)
