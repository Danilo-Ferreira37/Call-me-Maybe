*This project has been created as part of the 42 curriculum by dosorio-.*

# Call Me Maybe

## Description

**Call Me Maybe** is a **constrained decoding engine for function calling**, built entirely on top of a raw language model (no external function-calling framework, no fine-tuning). Given a JSON file describing a set of available functions (name, description, parameters and return type) and a JSON file containing a list of natural-language prompts, the program uses an LLM's raw next-token logits to:

1. Decide **which function** best matches each prompt.
2. Extract **the value of each parameter** required by that function, directly from the natural-language prompt.
3. Emit a single, well-formed JSON array containing, for every prompt, the original prompt, the chosen function name, and the extracted parameters — ready to be executed by any downstream tool.

The core idea is **constrained decoding**: instead of letting the model freely generate arbitrary text and hoping the output parses as valid JSON, the token stream is built manually, token by token. At every position, the set of tokens the model is allowed to pick from is restricted to only the tokens that keep the output syntactically valid (JSON punctuation, digits, `true`/`false`, one of the known function names, etc.). This guarantees the final output is always valid JSON, while still letting the LLM make the actual reasoning decisions (which function to call, and what value to extract for each parameter).

The project is a practical introduction to **function calling in LLMs**, implemented from first principles using only raw logits, without relying on any provider's built-in "tools" API.

---

## Instructions

### Requirements

- A Unix-like system (Linux/macOS)
- Python `>=3.12,<3.15`
- [`uv`](https://docs.astral.sh/uv/) — used to manage the virtual environment and dependencies
- Access to the 42 `llm_sdk` package (local dependency, declared as a path source in `pyproject.toml`)

### Installing `uv`

If you don't have `uv` installed yet, install it with:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your shell (or `source` your shell profile) so the `uv` command is available on your `PATH`.

### Installing dependencies

The project uses a `Makefile` that wraps `uv` for environment setup:

```sh
make install
```

This will:
- Set up the model cache directory under `$HOME/sgoinfre/.cache` (and symlink `$HOME/.cache` to it, migrating any existing cache), which is required so the downloaded LLM weights don't fill up your home quota on the school machines.
- Create a `.venv` virtual environment and install all dependencies declared in `pyproject.toml` via `uv sync`, if it doesn't already exist.

### Running the project

```sh
make run
```

By default this executes:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

You can also run it manually with custom input/output files:

```sh
uv run python -m src \
  --functions_definition path/to/your_functions.json \
  --input path/to/your_prompts.json \
  --output path/to/your_output.json
```

The result is written to `data/output/function_calling_results.json`.

### Other Make targets

| Command | Description |
|---|---|
| `make install` | Install project dependencies using `uv`. |
| `make run` | Execute the main script. |
| `make debug` | Run the main script with Python's built-in debugger (`pdb`). |
| `make lint` | Run `flake8` and `mypy` (relaxed rules). |
| `make lint-strict` | Run `flake8` and `mypy --strict`. |
| `make clean` | Remove caches (`__pycache__`, `.mypy_cache`) and generated output. |
| `make fclean` | Full cleanup: caches, output, vocab file, and the virtual environment. |
| `make help` | List all available commands. |

---

## Example Usage

**Function definitions** (`functions_definition.json`):

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {
      "name": { "type": "string" }
    },
    "returns": { "type": "string" }
  }
]
```

**Prompts** (`function_calling_tests.json`):

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Greet shrek" }
]
```

**Command:**

```sh
uv run python -m src \
  --functions_definition functions_definition.json \
  --input function_calling_tests.json \
  --output data/output/function_calls.json
```

**Resulting output** (`data/output/function_calls.json`):

```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2.0,
            "b": 3.0
        }
    },
    {
        "prompt": "Greet shrek",
        "name": "fn_greet",
        "parameters": {
            "name": "shrek"
        }
    }
]
```

A more complete example, including string extraction with regex-like reasoning:

```json
{
  "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
  "name": "fn_substitute_string_with_regex",
  "parameters": {
    "source_string": "Hello 34 I'm 233 years old",
    "regex": "\\d+",
    "replacement": "NUMBERS"
  }
}
```

---

## Algorithm Explanation

The core of the project is `ConstrainedDecoding.flow_decoding`, which builds a valid JSON object **token by token**, alternating between two modes:

1. **Free decoding under a restricted vocabulary** — used for structural JSON tokens (`{`, `}`, `:`, `,`, quotes, digits...) where the model has no real decision to make; the token is simply forced or picked from a tiny, pre-restricted set of candidates via `restricted_tokens`, which sets the logits of every disallowed token to `-inf` before taking the `argmax`.
2. **Guided sub-generation with a dedicated prompt** — used whenever the model actually has to *reason*: choosing a function name, or extracting a parameter value from natural language.

The full flow, in order:

1. **Prompt echo** — the literal user prompt is copied (encoded) directly into the output's `"prompt"` field. No generation needed here, since the value is already known — this avoids wasting model calls and guarantees a perfect, lossless copy.

2. **Function selection** — a **separate, isolated context** is built (not the JSON being generated) that lists every candidate function as `- name: description`, followed by the user's request and an instruction to pick the single best match. Instead of letting the model free-generate the name (which could hallucinate a name that doesn't exist, or drift mid-generation), selection is done **token-by-token with candidate elimination**:
   - At each position `i`, only the `i`-th token of each remaining candidate name is a legal choice.
   - The highest-logit legal token is picked; every candidate whose `i`-th token doesn't match is eliminated.
   - This repeats until exactly one candidate remains.
   
   This guarantees the generated name is **always** one of the valid function names — it is architecturally impossible for the model to hallucinate a nonexistent function.

3. **Parameter extraction** — once the function is fixed, its parameters are generated one at a time, dispatched by declared `type`:
   - **`boolean`** — the vocabulary is restricted to exactly the `true`/`false` tokens; a single `argmax` over these two logits is guaranteed to produce valid JSON.
   - **`number`** — a small dedicated prompt asks for the literal numeric value; generation is restricted to digit tokens (`0`–`9`) and the decimal point, and a trailing `.0` is appended automatically if the model stopped without emitting a fractional part, guaranteeing valid JSON float syntax.
   - **`string`** — a **few-shot prompt** is built containing the function's own name/description/parameter schema plus two worked examples (one about copying literal substrings without truncation, one about disambiguating between multiple similar candidate words in the sentence), followed by the actual question. Generation runs token-by-token until a closing `"` is produced by the model itself, bounded by a safety cap (`tokns_security`) to guarantee termination even if the model never emits a closing quote.
   - Any other/unknown type falls back to a generic literal-extraction prompt.

4. **Global safety bound** — the outer loop is bounded by `max_news_tokens`, and every inner extraction loop has its own token cap, so the algorithm is guaranteed to terminate even in worst-case model behavior (e.g. runaway repetition).

At the end, the manually assembled `input_ids` list is decoded back into text and concatenated into the final JSON array, which is then parsed once (as a final validation pass) before being written to disk.

---

## Design Decisions

- **Manual token assembly instead of free generation + regex/JSON-repair.** Post-hoc JSON repair is fragile and silently "fixes" malformed output in unpredictable ways. Building the token sequence manually, restricting the vocabulary at each step, guarantees **structural correctness by construction** rather than by luck.
- **Candidate elimination for function names**, instead of free generation of the name string. This closes off an entire class of hallucination bugs (the model inventing a function that was never declared) at the algorithm level rather than trying to catch it after the fact.
- **Separate "reasoning contexts" per decision.** The prompt used to pick a function name, and the prompt used to extract each parameter, are built from scratch rather than reusing the partially-built JSON output as context. This keeps each sub-decision focused and free of JSON syntax noise that would otherwise distract the model.
- **Few-shot examples for string extraction.** Early iterations used a plain instruction ("extract the literal value of parameter X"), which led the model to truncate values, resolve template placeholders it should have left untouched, or pick the wrong candidate word when several similar words appeared in the prompt. Adding two worked examples directly in the extraction context (one for verbatim copying, one for disambiguation) measurably improved accuracy — see [Challenges Faced](#challenges-faced).
- **Explicit delimiters that avoid collision with the content being extracted.** Using the literal `"` character both as a JSON delimiter *and* as a stop condition breaks down as soon as the extracted value itself contains a quote (e.g. `Say "hello" to {name}`). The extraction loop's stop condition and the escaping of the copied prompt text (`\\` and `"` are escaped before being re-encoded into the output) were designed specifically to avoid this class of ambiguity.
- **Hard iteration caps everywhere.** Every `while` loop that depends on the model choosing to emit a specific token (a closing quote, for example) has a numeric safety cap. Constrained decoding removes many failure modes, but it cannot remove the risk of the model never emitting the expected "stop" token — a bound guarantees the program always terminates.
- **A single final `json.loads` validation pass.** Even though the algorithm is designed to always emit valid JSON, the final assembled string is parsed once before being written to disk, so any residual malformed output fails loudly instead of producing a corrupted file.

---

## Performance Analysis

- **Accuracy — function selection:** very high. Because selection is done via token-by-token candidate elimination rather than free text generation, the function name in the output is *always* one of the declared functions; there is no way for this step to produce an invalid or hallucinated name. Misclassification can still happen (i.e. picking the wrong *valid* function when descriptions overlap semantically), but this was significantly reduced by clearly delimiting each function's `name: description` pair on its own line and adding an explicit instruction to reason about behavior rather than surface keywords.
- **Accuracy — numeric/boolean parameters:** effectively deterministic. Restricting the vocabulary to digits/point or to `true`/`false` removes almost all room for error; the only risk is the model reasoning about the *wrong* numeric value (e.g. confusing two numbers mentioned in the same prompt), not producing an invalid one.
- **Accuracy — string parameters:** the weakest link, since string values are open-ended and cannot be restricted to a small fixed vocabulary. Before the few-shot rework, common failure modes were: truncating literal values (dropping a leading `/` from a path), stopping generation early when the extracted value itself contained a `"`, and confusing two similarly-themed words in the same sentence (e.g. extracting a table name mentioned inside a SQL query instead of the actual database name). After adding structured few-shot examples and increasing the safety token cap for longer values, accuracy improved substantially, though it remains the part of the pipeline most sensitive to prompt phrasing.
- **Speed:** the engine trades raw throughput for reliability — it performs one forward pass per generated token, and additionally opens dedicated sub-contexts (with their own forward passes) for function selection and for every string parameter. This is noticeably slower than a single free-generation pass, but the trade-off is intentional: correctness and valid JSON output are prioritized over minimizing the number of model calls, since a single malformed token in an unconstrained approach can corrupt the entire output.
- **Reliability:** every loop is bounded, so the program cannot hang indefinitely regardless of model behavior, and the final `json.loads` check means the process fails fast and loudly rather than silently writing corrupted output.

---

## Challenges Faced

- **`.squeeze()` collapsing single-token tensors into scalars.** `tensor.squeeze()` on a tensor holding exactly one token returns a 0-dimensional tensor, and `.tolist()` on that returns a bare Python `int` instead of a `list`, breaking any subsequent `list.extend(...)` call with a `TypeError`. This surfaced repeatedly whenever a short piece of text (a function name, a fixed instruction string) happened to encode to a single token. It was fixed by switching to `.flatten().tolist()`, which always preserves at least one dimension and therefore always returns a list.
- **Unbounded generation loops.** Early versions of the string-extraction loop only stopped when the model emitted a closing `"`. Whenever the extraction prompt was ambiguous, the model would occasionally "hallucinate" an unrelated continuation (e.g. inventing a fake list of unrelated fields) and never emit the stop token, hanging indefinitely. This was solved by adding a hard token-count safety cap (`tokns_security`) to every extraction loop.
- **Delimiter collision.** Using `"` as both the JSON string delimiter and the generation stop condition breaks as soon as the value being extracted legitimately contains a quote character (e.g. `Say "hello" to {name}`), since the loop stops at the first `"` it sees, even mid-value. This required careful escaping of quotes/backslashes in the copied prompt text and careful phrasing of the extraction prompt.
- **Ambiguous or unstructured "reasoning" prompts.** The first version of the function-selection prompt concatenated all function name/description pairs with no clear separators, which made the model conflate the description of one function with a neighboring one — especially for functions covering semantically related domains (e.g. "square root" vs. "replace numbers in text", both involving numbers). This was fixed by formatting the function catalog as a clearly delimited, line-separated list, and by adding an explicit instruction to match based on *behavior*, not on incidental shared keywords.
- **Silent truncation of literal string values.** Even with a well-formed extraction prompt, the model would sometimes drop a leading character from a copied value (for example, extracting `home/user/data.json` instead of `/home/user/data.json`). This is a known bias of greedy decoding, where the first token of an unusual sequence is not always the model's top choice. Mitigations included reformatting the few-shot examples to mirror the exact sentence structure of the target case, and are documented as an area for further hardening (e.g. validating the extracted substring against the original prompt and re-attaching a missing prefix automatically).

---

## Testing Strategy

The implementation was validated primarily through **targeted example-driven testing**, using hand-crafted `functions_definition.json` / `function_calling_tests.json` pairs designed to isolate specific behaviors:

- **Type coverage:** dedicated test prompts for each supported parameter type (`number`, `integer`, `string`, `boolean`) to verify each extraction branch independently.
- **Function disambiguation:** prompts using vocabulary that overlaps between two or more semantically different functions (e.g. multiple functions involving "numbers"), to verify the selection step chooses based on intent rather than superficial keyword overlap.
- **String edge cases:** prompts specifically designed to stress the string-extraction path — values containing embedded quotes, values containing template-style placeholders (`{name}`) that must be preserved verbatim rather than resolved, long values close to the safety token cap, file paths with leading slashes/backslashes, and prompts with multiple plausible candidate words for the same parameter (e.g. a database name appearing alongside an unrelated table name in the same sentence).
- **Output validation:** every run's final assembled JSON string is parsed with `json.loads` as a hard pass/fail check — any malformed output causes the program to print the parsing error and exit non-zero (`exit(1)`), rather than producing a silently broken file.
- **Static analysis:** `make lint` / `make lint-strict` run `flake8` and `mypy` (including strict mode) over the whole codebase as part of the validation process, catching type inconsistencies early — particularly relevant given the amount of manual list/tensor manipulation involved in token handling.
- **Manual inspection of intermediate output**, using debug `print` statements on partially-built `input_ids` and on the sub-context prompts fed to the model, to trace exactly where a wrong token was chosen when a test case failed — this was the primary tool used to diagnose and fix the issues listed under [Challenges Faced](#challenges-faced).

---

## Resources

### Classic references

- https://www.youtube.com/watch?v=LPZh9BOjkQs
- https://pydantic.dev/docs/validation/latest/get-started/
- 
- 

### Use of AI

AI was used to help understand how LLMs work, to consult Python documentation like uv, to adjust the code to the mypy norm, and to help structure this README.