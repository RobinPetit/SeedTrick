# https://www.ncbi.nlm.nih.gov/nuccore/NM_005322
_hist1h1b_human =  'MSETAPAETATPAPVEKSPAKKKATKKAAGAGAAKRKATGPPVSELITKAVAASKERNGLSLAALKKALAAGGYDVEKNNSRIKLGLKSLVSKGTLVQTKGTGASGSFKLNKKAASGEAKPKAKKAGAAKAKKPAGATPKKAKKAAGAKKAVKKTPKKAKKPAAAGVKKVAKSPKKAKAAAKPKKATKSPAKPKAVKPKAAKPKAAKPKAAKPKAAKAKKAAAKKK'
# https://www.ncbi.nlm.nih.gov/nuccore/NM_020034
_hist1h1b_mouse = 'MSETAPAETAAPAPVEKSPAKKKTTKKAGAAKRKATGPPVSELITKAVSASKERGGVSLPALKKALAAGGYDVEKNNSRIKLGLKSLVSKGTLVQTKGTGASGSFKLNKKAASGEAKPKAKKTGAAKAKKPAGATPKKPKKTAGAKKTVKKTPKKAKKPAAAGVKKVAKSPKKAKAAAKPKKAAKSPAKPKAVKSKASKPKVTKPKTAKPKAAKAKKAVSKKK'

def test_imports():
    import seedtrick

def test_suffix_tree():
    K = 3
    from seedtrick.algo import suffixtree
    hist1h1b_human = _hist1h1b_human[:10]
    hist1h1b_mouse = _hist1h1b_mouse[:10]
    print(hist1h1b_human, hist1h1b_mouse, sep='\n')
    st = suffixtree.SuffixTree(hist1h1b_human, hist1h1b_mouse, K)
    for s in (hist1h1b_human, hist1h1b_mouse):
        for i in range(len(hist1h1b_mouse)-K+1):
            kmer = hist1h1b_mouse[i:i+K]
            expected = (hist1h1b_human.count(kmer), hist1h1b_mouse.count(kmer))
            found = st.count(kmer)
            assert found == expected
