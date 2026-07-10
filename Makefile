.VENV = .venv
PIP = $(.VENV)/bin/pip
RUN = uv run python -m 

run: install
	@$(RUN) src

install:
#	@mkdir -p $$HOME/sgoinfre/.cache
#
#	@if [ -d $$HOME/.cache ] && [ ! -L $$HOME/.cache ]; then \
#		cp -a $$HOME/.cache/. $$HOME/sgoinfre/.cache/; \
#		rm -rf $$HOME/.cache; \
#
#	@if [ ! -e $$HOME/.cache ]; then \
#		ln -s $$HOME/sgoinfre/.cache $$HOME/.cache; \
#	fi
	@test -d $(.VENV) || uv sync

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@clear
	@rm -f data/output/function_calling_results.json
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

debug: install
	@$(.VENV)/bin/python -m pdb -m src

lint: install
	@$(.VENV)/bin/flake8 .
	@$(.VENV)/bin/mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
	@echo "\033[32mEverything in the norm!!"

lint-strict: install
	@$(.VENV)/bin/flake8 .
	@$(.VENV)/bin/mypy . --strict
	@echo "\033[32mEverything in the norm!!"

help:
	@echo "\033[35mAvailable Make commands:\033[0m"
	@echo ""
	@echo "\033[33minstall\033[0m      Install project dependencies using uv."
	@echo "\033[33mrun\033[0m          Execute the main script project (e.g., via Python interpreter)."
	@echo "\033[33mdebug\033[0m        Run the main script in debug mode using Python’s built-in debugger (e.g., pdb)."
	@echo "\033[33mclean\033[0m        Remove temporary files or caches (e.g., __pycache__, .mypy_cache) to keep the project environment clean."
	@echo "\033[33mfclean\033[0m       Remove all temporary files, caches, and also delete the virtual environment."
	@echo "\033[33mlint\033[0m         Execute flake8 . and mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs"
	@echo "\033[33mlint-strict\033[0m  Execute flake8 . and mypy . --strict"

.DEFAULT:
	@echo "\033[31mError: Unknown command.\033[0m"
	@echo "Use \033[33mmake help\033[0m to see all available commands."

.PHONY: run debug clean fclean lint lint-strict help .DEFAULT
