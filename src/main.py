from src.parsing import read_input_files
from os import system
from src.constrained_decoding import ConstrainedDecoding
from llm_sdk import Small_LLM_Model


def main() -> None:
    func_def, func_call = read_input_files("data/input"
                                           "/functions_definition.json",
                                           "data/input"
                                           "/function_calling_tests.json")
    llm = Small_LLM_Model()
    system("clear")
    decoding = ConstrainedDecoding(llm, func_def, func_call)
    [decoding]
    # get_tokens_vocabulary()


if __name__ == "__main__":
    main()
