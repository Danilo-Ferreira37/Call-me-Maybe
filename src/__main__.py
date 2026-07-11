from src.parsing import read_input_files, parse_args
from os import system
import time
from src.constrained_decoding import ConstrainedDecoding
from llm_sdk import Small_LLM_Model


def main() -> None:
    """Load JSON input files, initialize the
    LLM and decoding engine, and
    execute the constrained decoding process."""

    start = time.time()
    args = parse_args()
    try:
        func_def, func_call = read_input_files(args.functions_definition,
                                               args.input)
    except FileNotFoundError:
        print("Error: FileNotFoundError")
    llm = Small_LLM_Model()
    system("clear")
    print("\033[34mCalling the Call-me-Maybe LLM...\033[0m")
    decoding = ConstrainedDecoding(llm, func_def, func_call, args.output)
    decoding.output_manager()
    end = time.time()
    print(f"\033[35m\nOutput file created at {args.output}\033[0m")
    print(f"\033[32mExecution time: {end - start:.2f} seconds\033[0m")

    # get_tokens_vocabulary()


if __name__ == "__main__":
    main()
