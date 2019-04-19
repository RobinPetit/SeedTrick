#cython:language_level=3

from seedtrick.svm.miksvm cimport MikSvm
from seedtrick.kernels.odh cimport ODHKernel
cimport numpy as np
import numpy as np
from libc.math cimport ceil, exp, sqrt

from scipy.sparse import lil_matrix

import cvxpy as cp

cdef inline double _squared_norm_ell2_sparse(object v):
    cdef np.ndarray data = np.asarray(v.data)
    return data.dot(data)

cdef np.ndarray pairwise_distances(object X, object Y):
    cdef int i, j
    if X is Y:
        print('Should optimize pairwise_distances to be more efficient if `X is Y`')
    ret = np.zeros((X.shape[0], Y.shape[0]), dtype=np.float)
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            tmp = X.getrowview(i)-Y.getrowview(j)
            ret[i,j] = _squared_norm_ell2_sparse(tmp)
    return ret

# TODO: optimize this a bit
cdef object kmeans_sparse(object samples, int n_clusters, int max_iter=20, bint verbose=False, bint heuristic=False):
    '''
    samples: np.ndarray (n_samples, n_features)
    '''
    assert isinstance(samples, lil_matrix)
    assert samples.shape[0] >= n_clusters
    cdef int n
    cdef np.ndarray clusters, previous_clusters
    if heuristic:
        n = <int>(sqrt(samples.shape[0]*n_clusters) + .5)
        samples = samples[np.random.choice(samples.shape[0], n, replace=False),:]
    clusters_indices = np.random.choice(samples.shape[0], size=n_clusters, replace=False)
    centroids = samples[clusters_indices]
    dists = pairwise_distances(centroids, samples)
    clusters = dists.argmin(axis=0)
    previous_clusters = clusters.copy()
    cdef bint loop = True
    cdef int iter_count = 0
    cdef np.ndarray nonzero_indices
    cdef int i, j, k
    while loop:
        if verbose:
            print('[kmeans] Iteration', iter_count)
        for i in range(n_clusters):  # ith cluster
            indices = np.where((clusters == i))[0]
            centroids.data[i] = [0.]*len(centroids.data[i])
            #centroids[i,centroids[i].nonzero()[1]] = 0
            for j in indices:  # jth sample related to ith cluster
                nonzero_indices = samples.getrowview(j).nonzero()[1]
                for k in range(len(nonzero_indices)):  # kth coordinate of jth sample
                    centroids[i, nonzero_indices[k]] += samples[j, nonzero_indices[k]]
            centroids.data[i] = list(np.asarray(centroids.data[i]) / len(indices))
        dists = pairwise_distances(centroids, samples)
        clusters = dists.argmin(axis=0)
        iter_count += 1
        loop = (clusters != previous_clusters).any() and iter_count < max_iter
    return clusters, centroids

cdef class Bag:
    cdef str bag
    cdef object odh_bag
    cdef np.ndarray indices
    cdef np.ndarray lengths
    cdef unsigned int nb_instances

    def __init__(self, object sequence, unsigned instance_size, bint shift):
        self.odh_bag = None
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
        self.odh_bag = odh.vectorize(self, lil=True)

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
    cdef object centroids
    cdef object kernels
    cdef object multi_kernel
    cdef np.ndarray subkernels
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
        cdef int i
        vector_X = self.odh.vectorize(X, lil=True)
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
        '''
        TODO: if at some point shogun stops producing segfaults...
        self.kernels = shogun.CombinedKernel()
        for i in range(self.nb_exp_points):
            self.kernels.append_kernel(shogun.CustomKernel(subkernels[i,...]))
        self.multi_kernel = shogun.MKLClassification(shogun.SVMLight())
        self.multi_kernel.set_mkl_norm(2)
        self.multi_kernel.parallel.set_num_threads(4)
        self.multi_kernel.set_kernel(self.kernels)
        self.multi_kernel.set_labels(shogun.BinaryLabels(y))
        self.multi_kernel.train()
        '''
        print('Optimizing')
        self._optimize(y)

    cdef void _compute_subkernels(self, Bag[:] bags):
        cdef int n_bags = bags.shape[0]
        cdef int i, j, k, ell, e, t, tmp
        cdef double prod
        cdef np.ndarray conformal_multipliers = np.empty((n_bags, self.nb_exp_points), dtype=object)
        cdef object bag_instances_i, bag_instances_j
        cdef object bag
        for i in range(n_bags):
            bag_instances_i = bags[i].get_bag()
            for e in range(self.nb_exp_points):
                conformal_multipliers[i,e] = list()
                for j in range(bag_instances_i.shape[0]):
                    conformal_multipliers[i,e].append(self._conformal_ker(bag_instances_i.getrowview(j), self.centroids.getrowview(e)))
        for i in range(n_bags):
            bag_instances_i = bags[i].get_bag()
            for j in range(i, n_bags):
                bag_instances_j = bags[j].get_bag()
                for k in range(bag_instances_i.shape[0]):
                    for ell in range(bag_instances_j.shape[0]):
                        prod = 0
                        for t in bag_instances_i.rows[k]:
                            prod += bag_instances_i[k,t]*bag_instances_j[ell,t]
                        if prod == 0:
                            continue
                        for e in range(self.nb_exp_points):
                            tmp = conformal_multipliers[i,e][k]*conformal_multipliers[j,e][ell]*prod
                            if tmp == 0:
                                continue
                            self.subkernels[e,i,j] += tmp
                            if i != j:
                                self.subkernels[e,j,i] += tmp
                self.subkernels[:,i,j] /= len(bags[i])*len(bags[j])

    cdef void _optimize(self, np.ndarray[np.int_t, ndim=1] y):
        cdef unsigned int n = y.shape[0]
        cdef np.ndarray weighted_kernels = y.reshape(-1, 1) * self.subkernels * y
        cdef np.ndarray traces = np.trace(self.subkernels, 1, 2)
        cdef unsigned int i
        _tau = cp.Parameter(nonneg=True)
        c = cp.Parameter(nonneg=True)
        _tau.value = self.tau
        c.value = traces.sum()
        zeta = cp.Variable(nonneg=True)
        alphas = cp.Variable(n, nonneg=True)
        obj = cp.Maximize(2*cp.sum(alphas) - _tau*cp.pnorm(alphas, p=2) - c*zeta)
        assert obj.is_dcp(), "objective is not DCP"
        constraints = [alphas <= self.C, alphas.T @ y == 0]
        eta_starting_idx = len(constraints)
        for i in range(self.nb_exp_points):
            constraints += [traces[i]*zeta >= cp.quad_form(alphas, weighted_kernels[i])]
        for i, c in enumerate(constraints):
            assert c.is_dcp(), "constraint {} is not DCP".format(c)
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

    cdef inline double rbf(self, double norm):
        norm /= 2*self.sigma*self.sigma
        return exp(-norm)

    cdef inline double _conformal_ker(self, object x, object c_e):
        return self.rbf(_squared_norm_ell2_sparse(x-c_e))
