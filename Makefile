PYTHON = python3
VENV = venv
PIP = $(VENV)/bin/pip
RUN = uv run

run:
	$(RUN) src/call_me_maybe.py

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@clear
	@echo "\033[32mEvery cleanup!!"

fclean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@rm -rf .venv
	@clear
	@echo "\033[32mProject full cleanup!!"

debug:
	$(RUN) -m pdb src/call_me_maybe.py

lint:
	$(VENV)/bin/flake8 .
	$(VENV)/bin/mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(VENV)/bin/flake8 .
	$(VENV)/bin/mypy . --strict

.PHONY: run debug clean fclean lint lint-strict
