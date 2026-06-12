from src.parsing import json_format_parse

def main():
    config = json_format_parse("data/input/function_calling_tests.json")

if __name__ == "__main__":
    main()
    #json_format_parse("data/input/functions_definition.json")