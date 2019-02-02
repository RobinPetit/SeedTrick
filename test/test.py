def test_imports():
    import seedtrick

def test_suffix_tree():
    from seedtrick.algo import suffixtree
    st = suffixtree.SuffixTree('banana')
    st.display()
