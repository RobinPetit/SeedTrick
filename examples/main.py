#!/usr/bin/env python3
from sys import argv

import sklearn.svm as svm
from sklearn.model_selection import train_test_split, LeaveOneOut
import numpy as np

import kernels as kr
from dataset import *
import test

def train_test_split_dict(X, y, test_size):
    assert set(X.keys()) == set(y.keys())
    keys = list(X.keys())
    xs = np.arange(len(keys))
    train_idx, test_idx = train_test_split(xs, xs, test_size=test_size)[:2]
    X_train = {keys[k]: X[keys[k]] for k in train_idx}
    y_train = {keys[k]: y[keys[k]] for k in train_idx}
    X_test = {keys[k]: X[keys[k]] for k in test_idx}
    y_test = {keys[k]: y[keys[k]] for k in test_idx}
    return X_train, X_test, y_train, y_test


class MultiInstanceSVC(svm.SVC):
    def __init__(self, kernel):
        self.svm_kernel = kr.SVMKernel(kernel)
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
    for K in (kr.StringSubsequenceKernel, kr.StringSubsequenceKernelDP):
        kernel = K(N=2, lambda_=lamb, normalized=normalized)
        k = kernel('car', 'cat')
        if normalized:
            expected = 1/(2 + lamb**2)
        else:
            expected = lamb**4
        print('k(\'cat\', \'car\'):', k)
        print('Expected:', expected)
        print('ratio:', k/expected)

def loo_dict(clf, X, y):
    assert isinstance(X, dict), str(type(X))
    N = len(X)
    predicted_accuracy = 0
    keys = list(X.keys())
    for i, k in enumerate(keys):
        value = X.pop(k)
        cls = y.pop(k)
        clf.fit(list(X.values()), list(y.values()))
        if clf.predict([value]) == cls:
            predicted_accuracy += 1
        X[k] = value
        y[k] = cls
    return predicted_accuracy / N

def get_best_clf(clf_generator, X, y):
    best_accuracy = -1
    best_clf = None
    for clf in clf_generator:
        accuracy = loo_dict(clf, X, y)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_clf = clf
    return best_clf, best_accuracy

def main():
    musk1, musk2 = load_musk_dataset()
    for i, data in enumerate((musk1, musk2)):
        X, y = data.X, data.y
        X_train, X_test, y_train, y_test = train_test_split_dict(X, y, test_size=.2)
        X_test = list(X_test.values())
        y_test = list(y_test.values())
        assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
        assert len(X_train) + len(X_test) == len(X)
        print(list(map(len, [X_train, X_test, y_train, y_test])))
        print('Dataset:', ['Musk1', 'Musk2'][i])
        for normalized in (False, True):
            print('\t[{}normalized]'.format('' if normalized else 'not '))
            generator_min_max_kernel = (MultiInstanceSVC(kr.MinMaxKernel(normalized, c, d)) for c in (0, 1e-2, 1e-1, 1.) for d in (1, 1.5, 2))
            generator_rbf_set_kernel = (MultiInstanceSVC(kr.RBFSetKernel(normalized, gamma=g)) for g in (.1, 1e-3, 1e-5, 1e-7))
            generators = [generator_min_max_kernel, generator_rbf_set_kernel]
            for j, g in enumerate(generators):
                print('\t\tKernel:', ['MinMax', 'SetKernel[RBF]'][j])
                print('\t\t[Training + CV]')
                clf, loo_accuracy = get_best_clf(g, X_train, y_train)
                print('\t\t[Testing]')
                predictions = clf.predict(X_test)
                test_accuracy = (predictions == y_test).sum() / len(X_test)
                print('\t\t\tKernel params:', clf.svm_kernel.get_kernel().get_params())
                print('\t\tAccuracy (LOO on train set): \033[1m{:3.2f}%\033[0m'.format(100*loo_accuracy))
                print('\t\tAccuracy (test set): \033[1m{:3.2f}%\033[0m'.format(100*test_accuracy))
        print('-'*40)

if __name__ == '__main__':
    if 'test-svm' in argv:
        test.test_svm()
    elif 'test-dataset' in argv or 'test-datasets' in argv:
        test.test_datasets()
    else:
        main()
