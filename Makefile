.PHONY: test eval check

test:
	pytest -q

eval:
	python -m lab.eval --split dev

check:
	python -m lab.eval --split check
