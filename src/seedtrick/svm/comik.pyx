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

    def vectorize(self, ODHKernel odh):
        if not self.is_vectorized:
            self.odh_bag = odh.vectorize(self)
            self.is_vectorized = True

cdef Bag[:] baggify(np.ndarray sequences, unsigned int instance_size, bint shift):
    cdef np.ndarray ret = np.empty(sequences.shape[0], dtype=object)
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
    cdef np.ndarray alpha

    def __init__(self, unsigned int k, unsigned int segment_size=50, unsigned int nb_exp_points=10, float tau=1, float sigma=25, float C=1):
        self.nb_exp_points = nb_exp_points
        self.k = k
        self.segment_size = segment_size
        self.odh = ODHKernel(k, True, False, segment_size)
        self.tau = tau
        self.sigma = sigma
        self.C = C

    def fit(self, np.ndarray X, np.ndarray[np.int_t] y):
        '''
        X = residue sequences
        '''
        print('vectorizing')
        cdef SparseMatrix vector_X = self.odh.vectorize(X)
        print('done')
        # Find the centroids
        _, self.centroids = kmeans_sparse(samples=vector_X, n_clusters=self.nb_exp_points, max_iter=50, verbose=True, heuristic=True)
        # Split instances into bags
        bags = baggify(X, self.segment_size, True)
        for bag in bags:
            bag.vectorize(self.odh)
        # compute subkernels
        shape = (self.nb_exp_points, X.shape[0], X.shape[0])
        self.subkernels = np.zeros(shape, dtype=np.double)
        print('Computing subkernels')
        self._compute_subkernels(bags)
        print('Optimizing')
        self._optimize(y)

    def _optimize(self, np.ndarray[np.int_t, ndim=1] y):
        self._optimize_formulation_1(y)
        #self._optimize_formulation_2(y)

    cdef void _compute_kappa_tilde(self, Bag[:] X):
        cdef unsigned int i, j, e
        cdef unsigned int M = 0
        cdef unsigned int E = self.nb_exp_points
        cdef SparseMatrix rows_of_Xi
        for i in range(len(X)):
            if len(X[i].get_bag()) > M:
                M = len(X[i].get_bag())
        self.kappa_tildes = np.empty((len(X), M, E), dtype=np.float)
        for i in range(len(X)):
            rows_of_Xi = X[i].get_bag()
            for j in range(len(rows_of_Xi)):
                for e in range(E):
                    self.kappa_tildes[i,j,e] = rows_of_Xi[j].dist(self.centroids[e])
        self.kappa_tildes = np.exp(-self.kappa_tildes / (2*self.sigma*self.sigma))

    cdef void _compute_subkernels(self, Bag[:] X):
        cdef unsigned int e, i, j
        cdef unsigned int E = self.nb_exp_points
        cdef unsigned int N = len(X)
        cdef double divisor
        self._compute_kappa_tilde(X)
        for e in range(E):
            for i in range(N):
                for j in range(i, N):
                    self.subkernels[e,i,j] = self.__compute_subkernel(e, i, j, X[i], X[j])
                    if j != i:
                        self.subkernels[e,j,i] = self.subkernels[e,i,j]
        for e in range(E):
            for i in range(N):
                divisor = sqrt(self.subkernels[e,i,i])
                self.subkernels[e,i,:] /= divisor
                self.subkernels[e,:,i] /= divisor

    cdef inline float __compute_subkernel(self, unsigned int e, unsigned int i, unsigned int j, Bag X_i, Bag X_j):
        cdef float ret = 0.
        cdef SparseMatrix transformed_X_i = X_i.get_bag()
        cdef SparseMatrix transformed_X_j = X_j.get_bag()
        cdef np.ndarray odh_products = cdot_sparse_matrices(transformed_X_i, transformed_X_j)
        cdef SparseVector c_e = self.centroids.rows[e]
        for idx_i in range(transformed_X_i.length()):
            kappa_tilde_i = self.kappa_tildes[i,idx_i,e]
            for idx_j in range(transformed_X_j.length()):
                kappa_tilde_j = self.kappa_tildes[j,idx_j,e]
                ret += kappa_tilde_i * kappa_tilde_j * odh_products[idx_i, idx_j]
        return ret

    def _optimize_formulation_1(self, np.ndarray[np.int_t, ndim=1] y):
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
        print('Traces:')
        print(traces)
        cdef unsigned int i
        _tau = cp.Parameter(nonneg=True)
        c = cp.Parameter(nonneg=True)
        _tau.value = self.tau
        c.value = traces.sum()
        t = cp.Variable(nonneg=False)
        alphas = cp.Variable(n, nonneg=True)
        obj = cp.Maximize(2*cp.sum(alphas) - _tau*cp.pnorm(alphas, p=2) - c*t)
        assert obj.is_dcp(), "objective is not DCP"
        constraints = [alphas <= self.C, alphas.T @ y == 0]
        eta_starting_idx = len(constraints)
        for i in range(self.nb_exp_points):
            W = cp.quad_form(alphas, cp.Parameter(shape=weighted_kernels[i].shape, value=weighted_kernels[i], PSD=True))
            constraints += [traces[i]*t >= W]
            assert constraints[-1].is_dcp(), 'constraint {} is not DCP'.format(len(constraints))
        for i, c in enumerate(constraints):
            assert c.is_dcp(), "constraint {} is not DCP".format(i)
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(verbose=False)
        except cp.error.SolverError as e:
            print('Error while solving the problem...')
            print(e)
        self.subkernel_weights = np.array([c.dual_value.flatten()[0] for c in prob.constraints[eta_starting_idx:]])
        # renormalise weights
        self.subkernel_weights /= sqrt(self.subkernel_weights.dot(self.subkernel_weights))
        self.alpha = np.asarray(alphas.value)
        # TODO: Meh?
        print('theta:', self.subkernel_weights)
        print('alpha\' y:', self.alpha.dot(y))
        print('alpha:', self.alpha)
        print('b:', prob.constraints[1].dual_value)

    cdef void _optimize_formulation_2(self, np.ndarray[np.int_t, ndim=1] y):
        '''
        From [2]

        ---
        [2] Bach, F. R., Lanckriet, G. R., & Jordan, M. I. (2004, July).
        Multiple kernel learning, conic duality, and the SMO algorithm. In Proceedings of the twenty-first international conference on Machine learning (p. 6). ACM.
        '''
        cdef unsigned int n = y.shape[0]
        #cdef np.ndarray weighted_kernels = y.reshape(-1, 1) * self.subkernels * y
        cdef np.ndarray traces = np.trace(self.subkernels, 1, 2)
        c = cp.Parameter(nonneg=True)
        c.value = traces.sum()
        alpha = cp.Variable(n, nonneg=True)
        zeta = cp.Variable(nonneg=False)
        obj = cp.Minimize(zeta - 2*cp.sum(alpha))
        assert obj.is_dcp(), "Objective not dcp"
        constraints = [alpha <= self.C, alpha.T @ y == 0]
        for i in range(self.nb_exp_points):
            constraints.append(traces[i] * zeta >= cp.quad_form(alpha, np.diag(y).dot(self.subkernels[i].dot(np.diag(y)))))
        for i, c in constraints:
            assert c.is_dcp(), i

    cdef inline float rbf(self, float norm):
        norm /= 2*self.sigma*self.sigma
        return exp(-norm)

    cdef inline float _conformal_ker(self, SparseVector x, SparseVector c_e):
        return self.rbf(vectors_dist(x, c_e))
