# -*- coding: utf-8 -*-

import numpy as np
import random as ran
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.cluster.vq as vq
import numpy.linalg as la

class GMM():
    def __init__(self, n, gmms):
        self.K = len(gmms)
        self.n = n
        self._gmms = gmms
        self.data, self.real_labels = self.generate_data()
        self.kmeans_labels = None
        self.em_labels = [1 for i in range(n)]
        self.kmeans_centroids = None
        self.dim = 2
        self.em_gmms = []
        self.weights = None

    # 生成n个数据点,返回数据点的列表,gmms数据类型为[means,cov]
    def generate_data(self):
        data, labels = [], []
        K = len(self._gmms)
        for i in xrange(self.n):
            k = ran.randint(0, K - 1)
            gmm = self._gmms[k]
            x, y = npr.multivariate_normal(gmm[0], gmm[1]).T
            data.append([x, y])
            labels.append(k)
        return data, labels

    def draw(self):
        self.kmeans()
        self.em()
        plt.figure(1)
        plt.figure(2)
        plt.figure(3)
        color = "bgrcmykw"
        for i in xrange(self.K):
            x, y = [], []
            xx, yy = [], []
            xxx, yyy = [], []
            for j in xrange(self.n):
                if self.real_labels[j] == i:
                    x.append(self.data[j][0])
                    y.append(self.data[j][1])
                if self.kmeans_labels[j] == i:
                    xx.append(self.data[j][0])
                    yy.append(self.data[j][1])
                if self.em_labels[j] == i:
                    xxx.append(self.data[j][0])
                    yyy.append(self.data[j][1])
            plt.figure(1)
            plt.plot(x, y, color[i]+'x')
            plt.axis('equal')
            plt.figure(2)
            plt.plot(xx,yy,color[i]+'x')
            plt.axis('equal')
            plt.figure(3)
            plt.plot(xxx, yyy, color[i] + 'x')
            plt.axis('equal')
        plt.show()

    def kmeans(self, nsteps=100):
        (self.kmeans_centroids, self.kmeans_labels) \
            = vq.kmeans2(self.data, self.K, minit="points", iter=nsteps)


    def pdf(self, gmm, data):
        centroids = np.mat(gmm[0]).T
        X = np.mat(data).T
        sigma = np.array(gmm[1])
        coe = (2*np.pi)**(self.dim /2.) * (np.fabs(la.det(sigma)))**0.5

        return np.exp(-0.5 * np.dot(np.dot((X-centroids).T, np.mat(sigma).T**-1), (X-centroids))) / coe

    def _em_init(self):
        self.kmeans()
        self.weights = [0 for i in range(self.K)]
        comps = [[] for i in range(self.K)]
        for i in xrange(self.n):
            self.weights[self.kmeans_labels[i]] += 1
            comps[self.kmeans_labels[i]].append(self.data[i])
        for i in xrange(self.K):
            self.weights[i] /= (1.0*self.n)
            em_gmm = [self.kmeans_centroids[i], np.cov(np.array(comps[i]).T).tolist()]
            self.em_gmms.append(em_gmm)

    def em(self, nsteps=100):
        self._em_init()
        for l in range(nsteps):
            # E step
            q = np.zeros((self.K, self.n))
            for j in range(self.n):
                for i in range(self.K):

                    q[i, j] = self.weights[i] * self.pdf(self.em_gmms[i], self.data[j])
            q = q / np.sum(q,axis=0) # normalize the weights

            # M step
            N = np.sum(q,axis=1)
            for i in range(self.K):
                centroids = np.dot(q[i,:],self.data) / N[i]
                sigma = np.zeros((self.dim, self.dim))
                for j in range(self.n):
                   sigma += q[i,j] * np.outer(self.data[j] - centroids, self.data[j] - centroids)
                sigma = sigma / N[i]
                self.em_gmms[i] = [centroids,sigma] # update the normal with new parameters
                self.weights[i] = N[i] / np.sum(N) # normalize the new weights
        for i in xrange(self.n):
            max = -1
            for k in xrange(self.K):
                pdf = self.pdf(self.em_gmms[k], data=self.data[i])
                if pdf > max:
                    self.em_labels[i] = k
                    max = pdf

if __name__ == '__main__':
    gmm1 = [[0,0], [[1,0],[0,2]]]
    gmm2 = [[0,-4], [[2,0], [0,1]]]
    gmms = [gmm1, gmm2]
    gmm = GMM(500,gmms)
    gmm.draw()