VENV = venv
PIP = $(VENV)/bin/pip
RUN = uv run python -m 

run:
	@$(RUN) src.main

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
	@rm data/vocab.json
	@echo "\033[32mProject full cleanup!!"

debug:
	#tenho que finalizar[[[]]]
	$(RUN) pdb src.main

lint:
	$(VENV)/bin/flake8 .
	$(VENV)/bin/mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(VENV)/bin/flake8 .
	$(VENV)/bin/mypy . --strict

.PHONY: run debug clean fclean lint lint-strict
