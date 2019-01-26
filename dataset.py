from collections import defaultdict
import numpy as np

DATA_PATH = '../data/'

class Dataset:
    def __init__(self, X, y, classification=None):
        '''
        Args:
            X:
                samples
            y:
                expected outputs
            classification:
                True if classification dataset, False if regression and None if not declared
        '''
        self.X = X
        self.y = y
        self.classification = classification

class BagDataset(Dataset):
    def __init__(self, X, y, classification):
        Dataset.__init__(self, X, y, classification)
        for kx in self.X:
            assert len(self.X[kx].shape) == 2

    def get_positive_instances(self):
        return self.get_instances(1)

    def get_negative_instances(self):
        return self.get_instances(0)

    def get_instances(self, _y):
        return ((self.X[kx], self.y[ky]) for (kx, ky) in zip(self.X, self.y) if self.y[ky] == _y)

    def data(self):
        X, y = list(), list()
        for k in self.X:
            X.append(self.X[k])
            y.append(self.y[k])
        return X, np.array(y)

def load_musk_dataset():
    data = [None, None]
    for i in range(1, 3):
        X, y = defaultdict(list), defaultdict(list)
        with open(DATA_PATH + 'musk/clean{}.data'.format(i)) as f:
            data[i-1] = list(map(lambda s: s.strip()[:-1].split(','), f))
        for sample in data[i-1]:
            molecule = sample[0]
            conformation = sample[1]
            y[molecule].append(int(sample[-1]))
            X[molecule].append(list(map(float, sample[2:-1])))
        y = {k: int(any(v)) for (k, v) in y.items()}
        X = {k: np.array(v, dtype=np.float) for (k, v) in X.items()}
        data[i-1] = BagDataset(X, y, classification=True)
    return data
