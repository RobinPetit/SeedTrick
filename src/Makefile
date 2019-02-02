install: setup.py
	python3 setup.py install && echo ''

build: setup.py 
	python3 setup.py build_ext --inplace && echo ''

clean:
	rm -r build/

.PHONY: build install clean
