#!/usr/bin/env python3
from sys import argv

import sklearn.svm as svm
from sklearn.model_selection import train_test_split
import numpy as np

import kernels
from dataset import *
import test

class MultiInstanceSVC(svm.SVC):
    def __init__(self, kernel):
        self.svm_kernel = kernels.SVMKernel(kernel)
        svm.SVC.__init__(self, kernel='precomputed')
        self.X_fit = None

    def fit(self, X, y):
        self.X_fit = X
        tmp = self.svm_kernel(X, X)
        svm.SVC.fit(self, tmp, y)

    def predict(self, X):
        tmp = self.svm_kernel(X, self.X_fit)
        return svm.SVC.predict(self, tmp)

    def score(self, X, y):
        return NotImplemented

def test_string_kernel():
    normalized = False
    lamb = .9
    for K in (kernels.StringSubsequenceKernel, kernels.StringSubsequenceKernelDP):
        kernel = K(N=2, lambda_=lamb, normalized=normalized)
        k = kernel('car', 'cat')
        if normalized:
            expected = 1/(2 + lamb**2)
        else:
            expected = lamb**4
        print('k(\'cat\', \'car\'):', k)
        print('Expected:', expected)
        print('ratio:', k/expected)

def main():
    musk1, musk2 = load_musk_dataset()
    for i, data in enumerate((musk1, musk2)):
        X, y = data.data()
        for x in X:
            assert len(x.shape) == 2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print('Dataset:', ['Musk1', 'Musk2'][i])
        for j, k in enumerate((kernels.MinMaxKernel(1, 2), kernels.RBFSetKernel(gamma=.1))):
            #for j, k in enumerate((kernels.MinMaxKernel(1, 2), kernels.SetKernel(kernels.RBFKernel(gamma=.1)))):
            print('\tKernel:', ['MinMax', 'SetKernel[RBF]'][j])
            clf = MultiInstanceSVC(kernel=k)
            clf.fit(X_train, y_train)
            print('\tAccuracy: {:3.2f}%'.format(100*(clf.predict(X_test) == y_test).sum() / len(y_test)))
        print('-'*25)

if __name__ == '__main__':
    if 'test-svm' in argv:
        test.test_svm()
    elif 'test-dataset' in argv or 'test-datasets' in argv:
        test.test_datasets()
    else:
        main()
