build: setup.py *.py
	python3 setup.py build_ext --inplace

.PHONY: build
