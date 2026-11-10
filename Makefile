PY=.venv/bin/python
setup:
	/opt/homebrew/bin/python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt
test:
	PYTHONPATH=. $(PY) -m pytest -q tests
run:
	PYTHONPATH=. $(PY) scripts/run_sim.py | tee logs/run_sim.log
