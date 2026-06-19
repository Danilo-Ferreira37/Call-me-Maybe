from src.parsing import read_input_files, tokens_vocabulary
from os import system
from src.constrained_decoding import ConstrainedDecoding
from llm_sdk import Small_LLM_Model


def main():
    system("clear")
    func_def, func_call = read_input_files("data/input/functions_definition.json", "data/input/function_calling_tests.json")
    tokens_vocabulary()

    llm = Small_LLM_Model()
    decoding = ConstrainedDecoding(llm, func_def, func_call)




if __name__ == "__main__":
    main()
