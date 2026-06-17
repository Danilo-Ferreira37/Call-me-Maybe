curl -LsSf https://astral.sh/uv/install.sh | sh

uv run python -m src [--functions_definition <function_definition_file>] [--input <input_file>] [--
output <output_file>]

uv run python -m src
--functions_definition data/input/functions_definition.json
--input data/input/function_calling_tests.json
--output data/output/function_calls.json




➜  Call-me-Maybe git:(main) ✗ grep -oP '"[^"]*":90,'  ~/sgoinfre/hf-cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/vocab.json
