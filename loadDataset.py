import pickle
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Returns a list of dictionaries, each with a 'data' key and a 'labels' key
def loadTrainingData():
    data_dicts = []
    for file in os.listdir("cifar-10-batches-py"):
        filename = os.fsdecode(file)
        if filename and filename[-1].isdigit():
            data_dicts.append(unpickle("cifar-10-batches-py/" + filename))
    return data_dicts

def loadMetaData():
    return unpickle("cifar-10-batches-py/batches.meta")

def loadTestData():
    return unpickle("cifar-10-batches-py/test_batch")