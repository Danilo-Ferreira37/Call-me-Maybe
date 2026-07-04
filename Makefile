.VENV = .venv
PIP = $(VENV)/bin/pip
RUN = uv run python -m 

run:
	test -d $(.VENV) || uv sync
	@$(RUN) src.main

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@clear
	@echo "\033[32mEverything cleanup!!"

fclean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@rm -rf .venv
	@rm -f data/vocab.json
	@rm -f data/output/function_calling_results.json
	@clear
	@echo "\033[32mProject full cleanup!!"

debug:
	#tenho que finalizar[[[]]]
	$(RUN) pdb src.main
	@echo "\033[32mEverything leanup!!"

lint:
	@$(.VENV)/bin/flake8 .
	@$(.VENV)/bin/mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
	@echo "\033[32mEverything in the norm!!"

lint-strict:
	@$(.VENV)/bin/flake8 .
	@$(.VENV)/bin/mypy . --strict
	@echo "\033[32mEverything in the norm!!"

.PHONY: run debug clean fclean lint lint-strict
