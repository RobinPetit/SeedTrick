from collections import defaultdict

# https://www.ncbi.nlm.nih.gov/nuccore/NM_005322
_hist1h1b_human =  'MSETAPAETATPAPVEKSPAKKKATKKAAGAGAAKRKATGPPVSELITKAVAASKERNGLSLAALKKALAAGGYDVEKNNSRIKLGLKSLVSKGTLVQTKGTGASGSFKLNKKAASGEAKPKAKKAGAAKAKKPAGATPKKAKKAAGAKKAVKKTPKKAKKPAAAGVKKVAKSPKKAKAAAKPKKATKSPAKPKAVKPKAAKPKAAKPKAAKPKAAKAKKAAAKKK'
# https://www.ncbi.nlm.nih.gov/nuccore/NM_020034
_hist1h1b_mouse = 'MSETAPAETAAPAPVEKSPAKKKTTKKAGAAKRKATGPPVSELITKAVSASKERGGVSLPALKKALAAGGYDVEKNNSRIKLGLKSLVSKGTLVQTKGTGASGSFKLNKKAASGEAKPKAKKTGAAKAKKPAGATPKKPKKTAGAKKTVKKTPKKAKKPAAAGVKKVAKSPKKAKAAAKPKKAAKSPAKPKAVKSKASKPKVTKPKTAKPKAAKAKKAVSKKK'

def _get_kmer_counts(s, K):
    ret = defaultdict(int)
    for i in range(len(s)-K+1):
        ret[s[i:i+K]] += 1
    return ret

def test_imports():
    import seedtrick

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
