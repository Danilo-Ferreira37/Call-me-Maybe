from src.parsing import read_input_files, parse_args
from os import system
from src.constrained_decoding import ConstrainedDecoding
from llm_sdk import Small_LLM_Model


def main() -> None:
    """Load JSON input files, initialize the
    LLM and decoding engine, and
    execute the constrained decoding process."""

    # Falta acabar o parsing, strings vazias, prompts ambiguos, parametros de definicao repetidos etc ...
    args = parse_args()
    try:
        func_def, func_call = read_input_files(args.functions_definition,
                                           args.input)
    except FileNotFoundError:
        print("Error: FileNotFoundError")
    llm = Small_LLM_Model()
    system("clear")
    decoding = ConstrainedDecoding(llm, func_def, func_call, args.output)
    decoding.output_manager()
    # get_tokens_vocabulary()


if __name__ == "__main__":
    main()
