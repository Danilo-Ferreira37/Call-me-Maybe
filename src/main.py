from src.parsing import parse_prompt, get_all_possible_tokens as get_token
from os import system
from src.constrained_decoding import ConstrainedDecoding
from llm_sdk import Small_LLM_Model


def main():
    system("clear")
    #config = parse_prompt("data/input/functions_definition.json")
    config = parse_prompt("data/input/function_calling_tests.json")
    get_token()

    llm = Small_LLM_Model()
    decoding = ConstrainedDecoding(llm, config)




if __name__ == "__main__":
    main()
