import multiprocessing
import numpy as np
import math

class kNN(object):

    def __init__(self, X, Y):
        # np array of images
        self.Xmodel = X
        # np array of image labels
        self.Ymodel = Y

    def calculateDistance(Xmodel, Ymodel, X, i, k, return_list):
        manhattanDists = np.sum(np.abs(Xmodel - X[i,:]), axis=1)
        closest = []
        closest.extend(np.argsort(manhattanDists)[:k])

        values = np.zeros((10))
        values[:] = np.nan
        for j in closest:
            if np.isnan(values[Ymodel[j]]):
                values[Ymodel[j]] = 1
            else:
                values[Ymodel[j]] += 1
        
        return_list[i] = np.nanargmax(values)
            

    def classify(self, X, k):
        manager = multiprocessing.Manager()
        num_test = X.shape[0]
        return_list = manager.list(range(num_test))

        pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
        for i in range(num_test):
            pool.apply_async(kNN.calculateDistance, args=(self.Xmodel, self.Ymodel, X, i, k, return_list))

        pool.close()
        pool.join()

        return return_list