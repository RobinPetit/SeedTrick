from collections import defaultdict
import numpy as np

# https://www.ncbi.nlm.nih.gov/nuccore/NM_005322
_hist1h1b_human =  'MSETAPAETATPAPVEKSPAKKKATKKAAGAGAAKRKATGPPVSELITKAVAASKERNGLSLAALKKALAAGGYDVEKNNSRIKLGLKSLVSKGTLVQTKGTGASGSFKLNKKAASGEAKPKAKKAGAAKAKKPAGATPKKAKKAAGAKKAVKKTPKKAKKPAAAGVKKVAKSPKKAKAAAKPKKATKSPAKPKAVKPKAAKPKAAKPKAAKPKAAKAKKAAAKKK'
# https://www.ncbi.nlm.nih.gov/nuccore/NM_020034
_hist1h1b_mouse = 'MSETAPAETAAPAPVEKSPAKKKTTKKAGAAKRKATGPPVSELITKAVSASKERGGVSLPALKKALAAGGYDVEKNNSRIKLGLKSLVSKGTLVQTKGTGASGSFKLNKKAASGEAKPKAKKTGAAKAKKPAGATPKKPKKTAGAKKTVKKTPKKAKKPAAAGVKKVAKSPKKAKAAAKPKKAAKSPAKPKAVKSKASKPKVTKPKTAKPKAAKAKKAVSKKK'

def _fasta_reader(path):
    ret = list()
    with open(path) as f:
        s = ''
        for l in f:
            if l.startswith('>'):
                if s == '':
                    continue
                ret.append(s)
                s = ''
            else:
                s += l.strip().upper()
        ret.append(s)
    return ret

def _get_kmer_counts(s, K):
    ret = defaultdict(int)
    for i in range(len(s)-K+1):
        ret[s[i:i+K]] += 1
    return ret

def test_imports():
    import seedtrick

def test_empty_sparse_vector():
    from seedtrick.algo.sparse import SparseVector
    v = SparseVector()
    assert len(v) == 0

def test_fill_sparse_vector():
    from seedtrick.algo.sparse import SparseVector
    v = SparseVector()
    for i in range(1, 6):
        v[2**i] = i
    assert len(v) == 5
    assert v[10] == 0
    assert v[16] == 4
    v[16] = 0
    assert len(v) == 4
    v = SparseVector()
    for i in range(5, 0, -1):
        v[2**i] = i
    assert len(v) == 5

def test_subtraction_sparse_vectors():
    from seedtrick.algo.sparse import SparseVector
    v = SparseVector()
    w = SparseVector()
    v[0] = v[3] = v[100] = 1
    w[0] = w[4] = 1
    z = v-w
    assert z[0] == 0
    assert z[3] == 1
    assert z[4] == -1
    assert z[100] == 1
    assert len(z) == 3

def test_add_sparase_vectors():
    from seedtrick.algo.sparse import SparseVector
    v = SparseVector()
    w = SparseVector()
    v[0] = v[3] = v[100] = 1
    w[0] = w[4] = 1
    z = v+w
    assert len(z) == 4
    assert z.dist(w+v) == 0

def test_dist_vectors():
    from seedtrick.algo.sparse import SparseVector
    v = SparseVector()
    w = SparseVector()
    v[0] = v[3] = v[100] = 1
    w[0] = w[4] = 1
    assert np.abs(v.dist(w) - np.sqrt(3)) < 1e-5  #32-bit floats vs numpy's 64-bit floats

def test_sparsevector_iadd():
    from seedtrick.algo.sparse import SparseVector
    v = SparseVector()
    v[0] = v[4] = 1
    v[5] = v[100] = 5
    w = SparseVector()
    w += v
    print(len(v), len(w))
    assert v.dist(w) == 0

def test_suffix_tree():
    K = 3
    from seedtrick.algo import suffixtree
    hist1h1b_human = _hist1h1b_human
    hist1h1b_mouse = _hist1h1b_mouse
    st = suffixtree.KmerSuffixTree(hist1h1b_human, hist1h1b_mouse, K)
    human_kmer_counts = _get_kmer_counts(hist1h1b_human, K)
    mouse_kmer_counts = _get_kmer_counts(hist1h1b_mouse, K)
    for s in (hist1h1b_human, hist1h1b_mouse):
        for i in range(len(hist1h1b_mouse)-K+1):
            kmer = s[i:i+K]
            expected = (human_kmer_counts.get(kmer, 0), mouse_kmer_counts.get(kmer, 0))
            found = st.count(kmer)
            assert found == expected

def test_spectrum_kernel():
    K = 3
    from seedtrick.kernels.spectrum import SpectrumKernel
    hist1h1b_human = _hist1h1b_human
    hist1h1b_mouse = _hist1h1b_mouse
    kernel = SpectrumKernel(K, normalized=False)
    kernel_value = kernel(hist1h1b_human, hist1h1b_mouse)
    human_kmer_counts = _get_kmer_counts(hist1h1b_human, K)
    mouse_kmer_counts = _get_kmer_counts(hist1h1b_mouse, K)
    expected_value = 0
    for k in set(human_kmer_counts.keys()) & set(mouse_kmer_counts.keys()):
        expected_value += human_kmer_counts[k] * mouse_kmer_counts[k]
    assert kernel_value == expected_value

def test_odh():
    from seedtrick.kernels import ODHKernel
    odh = ODHKernel(3, normalized=False, aa=True)
    X = ['CDEFG', 'ACDEFYRS']
    K = odh.get_K_matrix(X, X)
    print(K)
    assert K[0,1] == 3
    '''              ^
    common pairs of kmers are:
        CDE-CDE
        CDE-EDF
        EDF-EDF
    '''

def test_sparse_matrix_view():
    from seedtrick.algo.sparse import SparseMatrix
    M = SparseMatrix(1)
    M[0,7] = 10
    row = M[0]
    assert M[0,7] == row[7]
    row[5] = 14
    assert M[0, 5] == 14
    M2 = M[np.arange(1)]
    M2[0,5] = -2
    assert M[0,5] == M2[0,5]

def _test_mik():
    import numpy as np
    from seedtrick.svm import CoMIK
    DIR = '/media/robin/DATA/comik/sample_data/simulated_dataset1'
    pos_examples = _fasta_reader(DIR + '/pos.fasta')[:10]
    neg_examples = _fasta_reader(DIR + '/neg.fasta')[:10]
    assert set(''.join(pos_examples + neg_examples)) == set('ATCG')
    comik = CoMIK(2)
    print('{} pos examples and {} neg examples'.format(len(pos_examples), len(neg_examples)))
    X = np.array(pos_examples + neg_examples)
    y = np.zeros(len(X), dtype=np.int)
    y[:len(pos_examples)] = +1
    y[len(pos_examples):] = -1
    comik.fit(X, y)
