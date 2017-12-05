import kNN
import math
import random
import loadDataset
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt

def createModel():
    trainingData = loadDataset.loadTrainingData()

    data = []
    labels = []
    for dataDict in trainingData:
        data.extend(dataDict[b"data"])
        labels.extend(dataDict[b"labels"])

    dataNP = np.asarray(data)
    labelsNP = np.asarray(labels)

    return kNN.kNN(dataNP, labelsNP)

def main():
    test = loadDataset.loadTestData()
    metaData = loadDataset.loadMetaData()[b"label_names"]

    testLabels = test[b"labels"]
    testData = test[b"data"]

    cifarkNN = createModel()
    correct = 0
    number = math.floor(random.random() * len(testData))

    """
    for index,result in enumerate(cifarkNN.classify(testData[:100], 7)):
        if result == testLabels[index]:
            correct += 1

    print(correct/100 * 100)
    """

    result = cifarkNN.classify(testData[number:number+1], 7)

    img = np.reshape(np.reshape(testData[number:number+1], (3, 1024)).T, (32, 32, 3))
    plt.imshow(img)
    plt.title(metaData[result[0]].decode("utf-8"))
    plt.show()

if __name__ == "__main__":
    main()