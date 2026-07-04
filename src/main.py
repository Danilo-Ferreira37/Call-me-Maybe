from src.parsing import read_input_files
from os import system
from src.constrained_decoding import ConstrainedDecoding
from llm_sdk import Small_LLM_Model
# Nao se esquece de pegar as informações da forma certa:
# uv run python -m src [--functions_definition
# <function_definition_file>] [--input <input_file>]
# [--output <output_file>]


def main() -> None:
    """Load JSON input files, initialize the
    LLM and decoding engine, and
    execute the constrained decoding process."""
    func_def, func_call = read_input_files("data/input"
                                           "/functions_definition.json",
                                           "data/input"
                                           "/function_calling_tests.json")
    llm = Small_LLM_Model()
    system("clear")
    decoding = ConstrainedDecoding(llm, func_def, func_call)
    decoding.output_manager()
    # get_tokens_vocabulary()


if __name__ == "__main__":
    main()
