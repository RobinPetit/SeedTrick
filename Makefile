REMOTE_ORIGIN=git@github.com:RobinPetit/SeedTrick.git
DOCS_DIR=${HOME}/SeedTrick-doc/

install:
	make -C src/ install

build:
	make -C src/ build

doc: build
	export SPHINX_APIDOC_OPTIONS=members,show-inheritance,ignore-module-all && \
	sphinx-apidoc -Mef -o doc/source/SeedTrick src/seedtrick/ src/seedtrick/setup.py
	make -C doc/ html

pushdoc: doc
	cd ${DOCS_DIR}/html && \
	if [ ! -d ".git" ]; then git init; git remote add origin ${REMOTE_ORIGIN}; fi && \
	git add . && \
	git commit -m "Build the doc" && \
	git push -f origin HEAD:gh-pages

test: install
	pytest test/test.py

clean:
	make -C src/ clean
	make -C doc/ clean

.PHONY: build doc pushdoc test clean
