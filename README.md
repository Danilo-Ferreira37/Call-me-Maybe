*This project has been created as part of the 42 curriculum by dosorio-.*

# Call Me Maybe

## Description

**Call Me Maybe** is an educational project focused on introducing **function calling in Large Language Models (LLMs)** through a constrained decoding approach.

The main goal is to build a reliable pipeline where a language model can decide when and how to call predefined tools/functions, while respecting a strict output format and reducing invalid generations.  
This repository contains the core implementation, configuration files, and supporting assets to experiment with and evaluate this behavior.

In short, the project explores:
- structured tool/function invocation with LLMs;
- constrained decoding to enforce valid outputs;
- practical trade-offs between correctness, latency, and flexibility.

---

## Instructions

### 1) Clone the repository

```bash
git clone https://github.com/Danilo-Ferreira37/Call-me-Maybe.git
cd Call-me-Maybe
```

### 2) Install `uv` (if you do not have it)

On macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your shell (or source your shell config) and confirm:

```bash
uv --version
```

### 3) Install project dependencies

```bash
uv sync
```

### 4) Run the project

If your project exposes a package entrypoint in `src`, run with:

```bash
uv run python -m src
```

If you have a specific script (replace as needed), use:

```bash
uv run python src/main.py
```

### 5) Lint / type-check / quality (if configured)

```bash
uv run flake8 .
uv run mypy .
```

### 6) Optional: build helpers via Makefile

```bash
make help
```

Use the available targets shown by `make help` (or inspect the Makefile).

---

## Algorithm Explanation (Constrained Decoding Approach)

The constrained decoding strategy is designed to ensure that model outputs remain valid for function calling.  
Instead of allowing fully free-form generation, decoding is guided by structural constraints at each step.

### High-level flow

1. **Prompt + tool schema definition**  
   The model receives system/task instructions and a list of callable tools with argument schemas.

2. **Constrained token selection**  
   During generation, only tokens compatible with the expected structure are accepted (for example, valid JSON/function-call fields at a given position).

3. **Incremental validation**  
   Partial generations are continuously checked against structural expectations (e.g., balanced braces, valid keys, proper argument shape).

4. **Function-call emission**  
   When a valid tool call is completed, it is emitted in machine-readable form.

5. **Execution + response integration**  
   The selected function is executed, results are fed back into the loop, and the model can continue reasoning or finalize.

### Why this helps

- Reduces malformed tool calls.
- Improves consistency with expected schemas.
- Increases reliability when integrating with external executors/APIs.

---

## Design Decisions

Key implementation choices include:

- **Python-first architecture**  
  Chosen for fast iteration and rich ecosystem for LLM tooling.

- **`uv` for dependency management**  
  Faster and deterministic workflows with lockfile support.

- **Clear separation of concerns**  
  Distinct layers for prompt/schema preparation, decoding constraints, execution, and evaluation.

- **Config-driven behavior**  
  Parameters and constraints are centralized to make experiments reproducible and tunable.

- **Static checks in development workflow**  
  `flake8` and `mypy` support code quality and reduce regressions.

---

## Performance Analysis

The solution should be evaluated across three axes:

### 1) Accuracy
- Percentage of generations that produce a correct function call.
- Argument-level correctness (right fields, types, and values).
- Task success rate end-to-end.

### 2) Speed
- Latency overhead introduced by constrained decoding versus unconstrained generation.
- Tool execution time contribution.
- End-to-end time per request.

### 3) Reliability
- Invalid output rate (malformed calls).
- Recovery behavior after partial/failed generations.
- Stability across different prompts and edge cases.

In practice, constrained decoding usually trades a small amount of speed for better structural correctness and predictable behavior.

---

## Challenges Faced

Common difficulties in this type of project:

- **Balancing strictness vs flexibility**  
  Overly strict constraints can block useful outputs; loose constraints increase invalid calls.

- **Schema drift**  
  Prompt examples, parser expectations, and runtime schemas can diverge over time.

- **Edge-case parsing**  
  Handling incomplete JSON, escaping, or unexpected token sequences robustly.

- **Debugging model behavior**  
  Failures may come from prompt design, decoding rules, schema mismatch, or executor assumptions.

### How these were handled

- Added clearer structural expectations in prompts.
- Kept schemas centralized and versioned.
- Used incremental validation and defensive parsing.
- Introduced repeatable tests and quality checks.

---

## Testing Strategy

Validation combines multiple levels:

1. **Unit tests**  
   For parser/validator components and constraint logic.

2. **Integration tests**  
   For end-to-end flow: prompt → generation → function call → execution → final answer.

3. **Edge-case tests**  
   Invalid tokens, malformed partial outputs, missing arguments, unknown function names.

4. **Regression tests**  
   Fixed historical failures to avoid reintroducing bugs.

5. **Static analysis**  
   Linting (`flake8`) and type-checking (`mypy`) as baseline quality gates.

---

## Example Usage

### Example A: run a standard execution path

```bash
uv run python -m src
```

### Example B: run a direct script entrypoint

```bash
uv run python src/main.py
```

### Example C: quality checks

```bash
uv run flake8 .
uv run mypy .
```

### Example D: Makefile workflow

```bash
make help
make test
```

(Use only targets that exist in your Makefile.)

---

## Resources

Classic references and useful materials:

- OpenAI / function calling concepts and structured outputs
- JSON Schema documentation
- Python `typing` and static analysis documentation
- Articles/tutorials on constrained decoding and grammar-guided generation
- Prompt engineering best practices for tool use

### AI Usage Disclosure

AI tools were used to support:
- README drafting and structure organization;
- wording refinements and documentation clarity;
- summarization of methodology and evaluation dimensions.

AI was **not** used as a blind replacement for implementation decisions; final technical choices, validation logic, and project integration were reviewed and adjusted by the project author.
