PYTHON ?= python3
.PHONY: test verify regenerate bootstrap-lock fetch-wheels install-locked paper paper-pinned

test:
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) -O -m unittest discover -s tests -v

verify:
	$(PYTHON) scripts/run_verifiers.py --check

regenerate:
	$(PYTHON) scripts/run_verifiers.py --write

bootstrap-lock:
	$(PYTHON) tools/freeze_dependencies.py --wheelhouse build/new-wheelhouse --lock build/new-requirements.lock

fetch-wheels:
	$(PYTHON) -m pip --isolated download --only-binary=:all: --require-hashes --no-deps --dest wheelhouse -r requirements.lock

install-locked:
	$(PYTHON) -m pip --isolated install --only-binary=:all: --no-index --find-links=wheelhouse --require-hashes -r requirements.lock
	$(PYTHON) -m pip check

paper:
	$(PYTHON) tools/build_paper.py

paper-pinned:
	$(PYTHON) tools/pinned_build.py --epoch "$$(git log -1 --format=%ct)"
