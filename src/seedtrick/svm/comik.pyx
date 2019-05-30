#cython:language_level=3

'''
NOTE: Currently code is gross to look at. Cleanup and improvements regarding efficiency will be done when code is functional
'''

from seedtrick.svm.miksvm cimport MikSvm
from seedtrick.kernels.odh cimport ODHKernel
cimport numpy as np
import numpy as np
from libc.math cimport ceil, exp, sqrt

from seedtrick.algo.sparse cimport SparseVector, SparseMatrix, clone_matrix, pairwise_distances, cdot_sparse_matrices, vectors_dist

import cvxpy as cp

cdef tuple kmeans_sparse(SparseMatrix samples, int n_clusters, int max_iter=20, bint verbose=False, bint heuristic=False):
    '''
    samples: np.ndarray (n_samples, n_features)
    '''
    assert samples.length() >= n_clusters
    cdef int n
    if heuristic:
        n = <int>(sqrt(len(samples)*n_clusters) + .5)
        samples = samples[np.random.choice(len(samples), n, replace=False)]
    clusters_indices = np.random.choice(len(samples), size=n_clusters, replace=False)
    cdef SparseMatrix centroids = clone_matrix(samples[clusters_indices])
    cdef np.ndarray dists = pairwise_distances(centroids, samples)
    cdef np.ndarray clusters = dists.argmin(axis=0)
    cdef np.ndarray previous_clusters = clusters.copy()
    cdef bint loop = True
    cdef int iter_count = 0
    cdef np.ndarray nonzero_indices
    cdef int i, j, k
    cdef SparseVector ith_centroid
    while loop:
        if verbose:
            print('[kmeans] Iteration', iter_count)
        for i in range(n_clusters):  # ith cluster
            indices = np.where((clusters == i))[0]
            assert len(indices) > 0
            ith_centroid = centroids.rows[i]
            ith_centroid.flush()
            for j in indices:
                ith_centroid.plus_equal(samples[j])
            ith_centroid.divide_by_scalar(len(indices))
        dists = pairwise_distances(centroids, samples)
        clusters = dists.argmin(axis=0)
        iter_count += 1
        loop = (clusters != previous_clusters).any() and iter_count < max_iter
    return clusters, centroids

cdef np.ndarray _sigmoid(np.ndarray X):
    cdef np.ndarray div = 1 + np.exp(-X)
    return 1. / div

cdef class Bag:
    cdef str bag
    cdef SparseMatrix odh_bag
    cdef bint is_vectorized
    cdef np.ndarray indices
    cdef np.ndarray lengths
    cdef unsigned int nb_instances

    def __init__(self, object sequence, unsigned instance_size, bint shift):
        self.is_vectorized = False
        cdef int nb_chars = len(sequence)
        cdef unsigned int nb_instances_non_shift = <unsigned int>ceil(nb_chars / instance_size)
        cdef unsigned int nb_instances_shift = 0
        self.bag = str(sequence)
        if shift:
            nb_instances_shift = <unsigned int>ceil((nb_chars - instance_size//2) / instance_size)
        self.nb_instances = nb_instances_shift + nb_instances_non_shift
        self.indices = np.empty(self.nb_instances, dtype=np.int)
        self.lengths = np.empty(self.nb_instances, dtype=np.int)
        cdef int idx = 0
        cdef int N = 0
        for i in range(nb_instances_non_shift):
            self.indices[i] = idx
            N = nb_chars - idx
            if N >= instance_size:
                self.lengths[i] = instance_size
                idx += instance_size
            else:
                self.lengths[i] = N
                idx += N
                assert i == nb_instances_non_shift-1
        if not shift:
            return
        idx = instance_size//2
        for i in range(nb_instances_shift):
            i += nb_instances_non_shift
            self.indices[i] = idx
            N = nb_chars - idx
            if N >= instance_size:
                self.lengths[i] = instance_size
                idx += instance_size
            else:
                self.lengths[i] = N
                idx += N
                assert i == nb_instances_non_shift-1+nb_instances_shift

    def instances(self):
        for i in range(self.odh_bag.shape[0]):
            yield self.odh_bag[i]

    def get_bag(self):
        return self.odh_bag

    def __iter__(self):
        for idx, length in zip(self.indices, self.lengths):
            yield self.bag[idx:idx+length]

    def __array__(self):
        return np.fromiter(self, dtype=np.str_)

    def __len__(self):
        return self.nb_instances

    def vectorize(self, ODHKernel odh, bint verbose=False):
        if not self.is_vectorized:
            self.odh_bag = odh.vectorize(self, verbose)
            self.is_vectorized = True

cdef np.ndarray baggify(np.ndarray sequences, unsigned int instance_size, bint shift):
    cdef np.ndarray ret = np.empty(sequences.shape[0], dtype=Bag)
    cdef int i
    for i in range(sequences.shape[0]):
        ret[i] = Bag(sequences[i], instance_size, shift)
    return ret

cdef class CoMIK(MikSvm):
    cdef unsigned int nb_exp_points
    cdef unsigned int k
    cdef unsigned int segment_size
    cdef ODHKernel odh
    cdef SparseMatrix centroids
    cdef np.ndarray subkernels
    cdef np.ndarray kappa_tildes
    cdef float tau
    cdef float sigma
    cdef float C
    cdef np.ndarray subkernel_weights
    cdef np.ndarray alpha_times_y
    cdef float b
    # Keep data from training step in memory
    cdef np.ndarray X_train
    cdef SparseMatrix projected_X_train
    cdef np.ndarray bags_X_train

    def __init__(self, unsigned int k, unsigned int segment_size=50, unsigned int nb_exp_points=10, float tau=10, float sigma=10, float C=1):
        self.nb_exp_points = nb_exp_points
        self.k = k
        self.segment_size = segment_size
        self.odh = ODHKernel(k, True, False, segment_size)
        self.tau = tau
        self.sigma = sigma
        self.C = C

    def fit(self, np.ndarray X, np.ndarray y):
        print('[Train]')
        self.X_train = X
        self.projected_X_train = self.odh.vectorize(X, verbose=True)
        # Find the centroids
        _, self.centroids = kmeans_sparse(samples=self.projected_X_train, n_clusters=self.nb_exp_points, max_iter=50, verbose=True, heuristic=True)
        # Split instances into bags
        self.bags_X_train = baggify(X, self.segment_size, True)
        for i, bag in enumerate(self.bags_X_train):
            print('\tVectorizing bag {}/{}'.format(i+1, len(self.bags_X_train)))
            bag.vectorize(self.odh, verbose=False)
        shape = (self.nb_exp_points, X.shape[0], X.shape[0])
        self.subkernels = np.zeros(shape, dtype=np.double)
        self.kappa_tildes = self.__create_kappa_tilde_array(self.bags_X_train)
        print('Computing subkernels')
        self._compute_subkernels_train(self.bags_X_train, self.subkernels, self.kappa_tildes)
        print('[Optimizing]')
        self._optimize(y.astype(np.int))

    cdef inline np.ndarray __create_kappa_tilde_array(self, np.ndarray bags):
        cdef unsigned int max_nb_instances = max(len(bags[i].get_bag()) for i in range(len(bags)))
        return np.empty((len(bags), max_nb_instances, self.nb_exp_points), dtype=np.float)

    def _optimize(self, np.ndarray[np.int_t, ndim=1] y):
        #self._optimize_formulation_1(y)
        self._optimize_formulation_2(y)

    cdef void _compute_kappa_tilde(self, Bag[:] X, np.ndarray[np.float_t, ndim=3] kappa_tildes):
        cdef unsigned int i, j, e
        cdef unsigned int E = self.nb_exp_points
        cdef SparseMatrix rows_of_Xi
        for i in range(len(X)):
            rows_of_Xi = X[i].get_bag()
            for j in range(len(rows_of_Xi)):
                for e in range(E):
                    kappa_tildes[i,j,e] = rows_of_Xi[j].dist(self.centroids[e])
        kappa_tildes[:,:,:] = np.exp(-kappa_tildes / (2*self.sigma*self.sigma))

    cdef void _compute_subkernels_train(self, Bag[:] X, np.ndarray[np.float_t, ndim=3] subkernels, np.ndarray[np.float_t, ndim=3] kappa_tildes):
        cdef unsigned int e, i, j
        cdef unsigned int E = self.nb_exp_points
        cdef unsigned int N = len(X)
        cdef double divisor
        self._compute_kappa_tilde(X, kappa_tildes)
        for e in range(E):
            for i in range(N):
                for j in range(i, N):
                    subkernels[e,i,j] = self.__compute_subkernel(e, i, j, X[i], X[j], kappa_tildes, kappa_tildes)
                    if j != i:
                        subkernels[e,j,i] = subkernels[e,i,j]
        self._normalise_kernels_train(subkernels)

    cdef void _normalise_kernels_train(self, np.ndarray[np.float_t, ndim=3] subkernels):
        cdef unsigned int e, i, j
        cdef unsigned int E = self.nb_exp_points
        cdef unsigned int nb_rows = subkernels.shape[1]
        cdef unsigned int nb_cols = subkernels.shape[2]
        cdef np.ndarray divisors_rows = np.empty(nb_rows, dtype=np.float)
        cdef np.ndarray divisors_cols = np.empty(nb_cols, dtype=np.float)
        for e in range(E):
            for i in range(nb_rows):
                divisors_rows[i] = sqrt(subkernels[e,i,i])
            for j in range(nb_cols):
                divisors_cols[j] = sqrt(subkernels[e,j,j])
            for i in range(nb_rows):
                subkernels[e,i,:] /= divisors_rows[i]
            for j in range(nb_cols):
                subkernels[e,:,j] /= divisors_cols[j]

    cdef inline float __compute_subkernel(self, unsigned int e, unsigned int i, unsigned int j, Bag X_i, Bag Y_j,
                                          np.ndarray[np.float_t, ndim=3] kappa_tildes_X, np.ndarray[np.float_t, ndim=3] kappa_tildes_Y):
        cdef float ret = 0.
        cdef unsigned int idx_i, idx_j
        cdef SparseMatrix transformed_X_i = X_i.get_bag()
        cdef SparseMatrix transformed_Y_j = Y_j.get_bag()
        cdef np.ndarray odh_products = cdot_sparse_matrices(transformed_X_i, transformed_Y_j)
        cdef SparseVector c_e = self.centroids.rows[e]
        cdef float kappa_tilde_X_i, kappa_tilde_Y_j
        for idx_i in range(transformed_X_i.length()):
            kappa_tilde_X_i = kappa_tildes_X[i,idx_i,e]
            for idx_j in range(transformed_Y_j.length()):
                kappa_tilde_Y_j = kappa_tildes_Y[j,idx_j,e]
                ret += kappa_tilde_X_i * kappa_tilde_Y_j * odh_products[idx_i, idx_j]
        return ret

    cdef void _optimize_formulation_1(self, np.ndarray[np.int_t, ndim=1] y):
        '''
        From [1]

        ---
        [1] Lanckriet, G. R., Cristianini, N., Bartlett, P., Ghaoui, L. E., & Jordan, M. I. (2004).
        Learning the kernel matrix with semidefinite programming. Journal of Machine learning research, 5(Jan), 27-72.
        '''
        cdef unsigned int n = y.shape[0]
        cdef np.ndarray weighted_kernels = y.reshape(-1, 1) * self.subkernels * y
        cdef np.ndarray traces = np.trace(self.subkernels, 1, 2)
        assert traces.shape[0] == self.nb_exp_points
        cdef unsigned int i
        _tau = cp.Parameter(nonneg=True)
        c = cp.Parameter(nonneg=True)
        _tau.value = self.tau
        c.value = 1  #traces.sum()
        t = cp.Variable(nonneg=False)
        alphas = cp.Variable(n, nonneg=True)
        obj = cp.Maximize(2*cp.sum(alphas) - _tau*cp.pnorm(alphas, p=2) - c*t)
        assert obj.is_dcp(), "objective is not DCP"
        constraints = [alphas <= self.C, alphas.T @ y == 0]
        for i in range(self.nb_exp_points):
            W = cp.quad_form(alphas, cp.Parameter(shape=weighted_kernels[i].shape, value=weighted_kernels[i], PSD=True))
            constraints += [traces[i]*t >= W]
            assert constraints[-1].is_dcp(), 'constraint {} is not DCP'.format(len(constraints))
        prob = cp.Problem(obj, constraints)
        prob.solve(verbose=False)
        print('alphas:', alphas.value)
        self.subkernel_weights = np.sqrt(np.array([c.dual_value.flatten()[0] for c in prob.constraints[2:]]))
        self.alpha_times_y = np.asarray(alphas.value) * y
        self.b = prob.constraints[1].dual_value

    cdef void _optimize_formulation_2(self, np.ndarray[np.int_t, ndim=1] y):
        '''
        From [2]

        ---
        [2] Bach, F. R., Lanckriet, G. R., & Jordan, M. I. (2004, July).
        Multiple kernel learning, conic duality, and the SMO algorithm. In Proceedings of the twenty-first international conference on Machine learning (p. 6). ACM.
        '''
        cdef unsigned int n = y.shape[0]
        #cdef np.ndarray weighted_kernels = y.reshape(-1, 1) * self.subkernels * y
        print(self.subkernels)
        cdef np.ndarray traces = np.trace(self.subkernels, 1, 2)
        c = cp.Parameter(nonneg=True)
        c.value = traces.sum()
        alpha = cp.Variable(n, nonneg=True)
        zeta = cp.Variable(nonneg=False)
        obj = cp.Minimize(zeta - 2*cp.sum(alpha))
        assert obj.is_dcp(), "Objective not dcp"
        constraints = [alpha <= self.C, alpha.T @ y == 0]
        for i in range(self.nb_exp_points):
            constraints.append(zeta * traces[i]/c >= cp.quad_form(alpha, np.diag(y).dot(self.subkernels[i].dot(np.diag(y)))))
            assert constraints[-1].is_dcp(), "Constraint {} is not DCP".format(len(constraints))
        prob = cp.Problem(obj, constraints)
        prob.solve(verbose=False)
        print('alphas:', alpha.value)
        self.subkernel_weights = np.sqrt(np.array([c.dual_value.flatten()[0] for c in prob.constraints[2:]]))
        assert self.subkernel_weights.shape[0] == self.nb_exp_points
        self.alpha_times_y = np.asarray(alpha.value) * y
        self.b = prob.constraints[1].dual_value

    def score(self, np.ndarray X, np.ndarray y):
        return (self.predict(X) == y).mean()

    def predict(self, np.ndarray X):
        probas = self.predict_probas(X)
        assert probas.shape[0] == X.shape[0]
        assert probas.ndim == 1
        ret = np.ones(probas.shape)
        ret[probas < .5] = -1
        return ret

    def predict_probas(self, np.ndarray X):
        return self._predict_probas(X)

    cdef np.ndarray _predict_probas(self, np.ndarray X_test):
        cdef np.ndarray bags_test = baggify(X_test, self.segment_size, True)
        for bag in bags_test:
            bag.vectorize(self.odh, verbose=False)
        cdef np.ndarray subkernels = np.empty((self.nb_exp_points, len(self.bags_X_train), len(bags_test)), dtype=np.float)
        self._compute_subkernels_test(self.bags_X_train, bags_test, subkernels)
        cdef np.ndarray gram_matrix = (subkernels * self.subkernel_weights.reshape(-1, 1, 1)).sum(axis=0)
        return _sigmoid(self.alpha_times_y.dot(gram_matrix) + self.b)

    cdef void _compute_subkernels_test(self, np.ndarray bags_train, np.ndarray bags_test, np.ndarray[np.float_t, ndim=3] subkernels):
        cdef unsigned int i, j, e
        cdef unsigned int E = self.nb_exp_points
        cdef np.ndarray kappa_tildes_X = self.__create_kappa_tilde_array(bags_train)
        cdef np.ndarray kappa_tildes_Y = self.__create_kappa_tilde_array(bags_test)
        self._compute_kappa_tilde(bags_train, kappa_tildes_X)
        self._compute_kappa_tilde(bags_test, kappa_tildes_Y)
        cdef np.ndarray normalisation_factor_train = np.empty(len(bags_train), dtype=np.float)
        cdef np.ndarray normalisation_factor_test = np.empty(len(bags_test), dtype=np.float)
        for e in range(E):
            for j in range(len(bags_test)):
                normalisation_factor_test[j] = self.__compute_subkernel(e, j, j, bags_test[j], bags_test[j], kappa_tildes_Y, kappa_tildes_Y)
            for i in range(len(bags_train)):
                normalisation_factor_train[i] = self.__compute_subkernel(e, i, i, bags_train[i], bags_train[i], kappa_tildes_X, kappa_tildes_X)
                for j in range(len(bags_test)):
                    subkernels[e,i,j] = self.__compute_subkernel(e, i, j, bags_train[i], bags_test[j], kappa_tildes_X, kappa_tildes_Y)
                    subkernels[e,i,j] /= sqrt(normalisation_factor_train[i] * normalisation_factor_test[j])
