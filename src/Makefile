install: setup.py
	python3 setup.py install && echo ''

build: setup.py 
	python3 setup.py build_ext --inplace && echo ''

clean:
	rm -r build/

rm:
	rm -f seedtrick/**/[!_]*.c

.PHONY: build install clean rm
