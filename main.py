#!/usr/bin/env python3
import sklearn.svm
import numpy as np
import kernels

import test

from sys import argv

X, y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([1, 0, 0, 1])

def main():
    normalized = False
    lamb = .9
    for K in (kernels.StringKernel, kernels.StringKernelDP):
        kernel = K(N=2, lambda_=lamb, normalized=normalized)
        k = kernel('car', 'cat')
        if normalized:
            expected = 1/(2 + lamb**2)
        else:
            expected = lamb**4
        print('k(\'cat\', \'car\'):', k)
        print('Expected:', expected)
        print('ratio:', k/expected)

if __name__ == '__main__':
    if 'test' in argv:
        test.test_svm()
    else:
        main()
