from src.parsing import FuncDef, FuncCall
import json
import os
from typing import TYPE_CHECKING, cast


if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


class ConstrainedDecoding:
    """Implements a constrained decoding engine that selects and executes the
    appropriate function call for each prompt. It encodes function metadata,
    restricts token choices during generation, and produces structured JSON
    output based on the model's logits."""
    def __init__(s, llm: "Small_LLM_Model", definition: list[FuncDef],
                 prompts: list[FuncCall], output_file: str) -> None:
        s.llm = llm
        s.prompts = prompts
        s.output_file = output_file
        s.context_file = json.dumps(
                                    [d.model_dump() for d in definition],
                                    ensure_ascii=False,
                                    indent=2)
        s.set_possible_tokens(definition)

    def set_possible_tokens(s, definition: list[FuncDef]) -> None:
        s.possible_names = []
        s.dscrp = []
        s.parameters = []

        for i, dct in enumerate(definition):
            parameter = {}

            s.possible_names.append(s.llm.encode(dct.name).squeeze().tolist())
            s.dscrp.append(s.llm.encode(dct.description).squeeze().tolist())
            for para_name, para_type in definition[i].parameters.items():
                parameter[para_name] = str(para_type).replace("type=",
                                                              "").replace("'",
                                                                          "")
            s.parameters.append(parameter)

    def restricted_tokens(s, tokens: list[int]) -> list[float]:
        restricted_list = [-float('inf')] * len(s.logits)
        for tk in tokens:
            restricted_list[tk] = s.logits[tk]
        return restricted_list

    def output_manager(s) -> None:
        s.output = "["
        for i in range(len(s.prompts)):
            s.prompt = s.prompts[i].prompt

            s.original_input = s.llm.encode(s.prompt).squeeze().tolist()
            s.flow_decoding()
            if i < len(s.prompts) - 1:
                s.output += ","
        s.output += "]"
        try:
            data = json.loads(s.output)
        except json.JSONDecodeError as e:
            print("Fatal Error:")
            print(e)
            exit(1)
        os.makedirs("data/output", exist_ok=True)
        with open("data/output/function_calling_results.json", "w") as f:
            json.dump(data, f, indent=4)

    def get_lgts(s, tokens_ids: list[int]) -> list[float]:
        # The cast function is only to the mypy checker
        return cast(list[float], s.llm.get_logits_from_input_ids(tokens_ids))

    def flow_decoding(s) -> None:
        # Tokens ID in JSON vocabulary
        open_chav = 90
        close_chav = 92
        true = 1866
        false = 3849
        space = 220
        comma = 11  # virgula
        colon = 25  # dois pontos
        prompt = 40581
        point = 13
        quotes = 1  # aspas
        name = 606
        zero = 15
        parameters = 13786

        input_ids = [open_chav, quotes, prompt, quotes, colon, space, quotes]
        count = 0
        max_news_tokens = 100
        wrote_name = False
        idx_func_name = 0

        while count < max_news_tokens:
            s.logits = s.get_lgts(input_ids)

            if wrote_name:
                add_context = ""
                para_count = 0
                for k, type_var in s.parameters[idx_func_name].items():
                    para_count += 1
                    variable = s.llm.encode(k).squeeze().tolist()

                    input_ids.append(quotes)
                    if isinstance(variable, int):
                        input_ids.append(variable)
                        input_ids.extend([quotes, colon, space])
                    else:
                        input_ids.extend(variable)
                        input_ids.extend([quotes, colon, space])

                    if type_var == "string":
                        params_schema = ", ".join(
                            [f"{pk}:{pv}" for
                             pk, pv in s.parameters[idx_func_name].items()]
                        )
                        rules = (
                                 'Extract parameter values EXACTLY as'
                                 ' they appear in the text. '
                                 'Copy the full literal string, never '
                                 'shorten or simplify it. '
                                 'When the text mentions multiple '
                                 'similar words, pick the one that '
                                 'matches the parameter\'s role in '
                                 'the sentence, not just any nearby word.\n\n'

                                 'Function=fn_read_file\n'
                                 'Description=Read a file from the given '
                                 'path with specified encoding.\n'
                                 'Params=path:string, encoding:string\n'
                                 'User="Read the file at '
                                 '/var/log/app/output.log with ascii '
                                 'encoding"\n'
                                 'Param="path" type=string value="'
                                 '/var/log/app/output.log"\n\n'

                                 'Function=fn_execute_sql_query\n'
                                 'Description=Execute a SQL query on'
                                 ' a specified database.\n'
                                 'Params=query:string, database:string\n'
                                 'User="Run the query \'DELETE'
                                 ' FROM sessions\' on the analytics '
                                 'database"\n'
                                 'Param="database" type=string'
                                 ' value="analytics"\n\n'
                                )
                        ctxt = (rules +
                                f"Function={s.llm.decode(s.
                                                         possible_names
                                                         [idx_func_name])}\n"
                                f"Description={s.llm.decode(s.dscrp
                                                            [idx_func_name]
                                                            )}\n"
                                f"Params={params_schema}\n"
                                f'User="{s.prompt.replace("\\", "\\\\")}"\n'
                                f'Param="{k}" type=string value="')
                        input_ids.append(quotes)
                        final_name = ""
                        tokns_security = 0
                        while tokns_security < 15:
                            # Context token == ctxt_tk
                            ctxt_tk = s.llm.encode(ctxt).squeeze().tolist()
                            ctxt_logits = s.get_lgts(ctxt_tk)
                            next_tk = ctxt_logits.index(max(ctxt_logits))
                            next_tk_str = s.llm.decode(next_tk)
                            if '"' in next_tk_str:
                                final_name += str(next_tk_str.
                                                  replace('"', "").
                                                  replace('\n', ""))
                                break
                            ctxt += s.llm.decode(next_tk)
                            final_name += next_tk_str
                            tokns_security += 1

                        fname_tk = s.llm.encode(final_name).squeeze().tolist()
                        if isinstance(fname_tk, int):
                            input_ids.append(fname_tk)
                        else:
                            input_ids.extend(fname_tk)

                        input_ids.append(quotes)

                    elif type_var == "boolean":
                        context_lgt = s.restricted_tokens([false, true])
                        input_ids.append(context_lgt.index(max(context_lgt)))

                    elif type_var == "number":
                        ctxt = str(f"Given the question {s.prompt},"
                                   f"{add_context} the "
                                   f"literal value of parameter '{k}' is ")
                        number = ""
                        while True:
                            ctxt_tk = s.llm.encode(ctxt).squeeze().tolist()
                            ctxt_lgt = s.get_lgts(ctxt_tk)
                            next_token = ctxt_lgt.index(max(ctxt_lgt))
                            if ((next_token >= 15 and next_token <= 24)
                               or next_token == point):
                                input_ids.append(next_token)
                                ctxt += s.llm.decode(next_token)
                                number += s.llm.decode(next_token)
                            else:
                                if "." not in number:
                                    number += ".0"
                                    input_ids.extend([point, zero])
                                elif number[-1] == ".":
                                    number += "0"
                                    input_ids.append(zero)
                                break
                        add_context += f"where {k} = {number}, "

                    else:
                        ctxt = str(f"Given the question {s.prompt},"
                                   f"{add_context} the value of "
                                   f"parameter '{k}' is ")
                        number = ""
                        while True:
                            cntxt_tk = s.llm.encode(ctxt).squeeze().tolist()
                            cntxt_logits = s.get_lgts(cntxt_tk)
                            next_token = cntxt_logits.index(max(cntxt_logits))
                            if (next_token >= 15 and next_token <= 24):
                                input_ids.append(next_token)
                                ctxt += s.llm.decode(next_token)
                                number += s.llm.decode(next_token)
                            else:
                                break
                        add_context += f"where {k} = {number}, "

                    if para_count == len(s.parameters[idx_func_name]):
                        input_ids.append(close_chav)
                    else:
                        input_ids.extend([comma, space])
                break

            elif input_ids[-1] == quotes and input_ids[-5] == prompt:
                str_input = s.llm.decode(s.original_input)
                str_input = str_input.replace("\\", "\\\\").replace('"',
                                                                    '\\"')
                tk = s.llm.encode(str_input).squeeze().tolist()
                input_ids.extend(tk)
                input_ids.extend([quotes, comma, space, quotes, name,
                                  quotes, colon, space, quotes])
                count += len(s.original_input) + 9
                continue

            elif input_ids[-5] == name:
                f_name = []
                cntxt: list[int] = []
                cntxt.extend(s.llm.encode("Available functions:"
                                          "\n").flatten().tolist())
                for i in range(len(s.possible_names)):
                    cntxt.extend(s.llm.encode("- ").flatten().tolist())
                    cntxt.extend(s.possible_names[i])
                    cntxt.extend([colon, space])
                    cntxt.extend(s.dscrp[i])
                    cntxt.extend(s.llm.encode("\n").flatten().tolist())
                cntxt.extend(s.llm.encode("\nUser "
                                          "request: ").squeeze().tolist())
                cntxt.extend(s.original_input)
                cntxt.extend(s.llm.encode(
                    "\nBased on the request "
                    "above, choose exactly one function name "
                    "from the list that best matches the task. "
                    "Do not choose based on "
                    "surface keywords alone — consider what the "
                    "function actually does.\n"
                    "Most appropriate function name: "
                ).flatten().tolist())
                candidate_names = s.possible_names.copy()
                i = 0
                while len(candidate_names) > 1:
                    s.logits = s.get_lgts(cntxt)
                    possibles_tk = [tk[i] for tk in candidate_names]
                    restricted_list = s.restricted_tokens(possibles_tk)
                    next_tk = restricted_list.index(max(restricted_list))
                    f_name.append(next_tk)
                    cntxt.append(next_tk)
                    for candidate in candidate_names.copy():
                        if next_tk != candidate[i]:
                            candidate_names.remove(candidate)
                    i += 1
                f_name = candidate_names[0]
                idx_func_name = s.possible_names.index(f_name)
                input_ids.extend(s.possible_names[idx_func_name])
                input_ids.extend([quotes, comma, space, quotes,
                                  parameters, quotes, colon, space,
                                  open_chav])
                count += len(s.possible_names[idx_func_name]) + 10
                wrote_name = True
                continue

            else:
                restricted_list = s.logits
            idx = restricted_list.index(max(restricted_list))
            input_ids.append(idx)

            count += 1
        input_ids.append(close_chav)
        s.output += s.llm.decode(input_ids)
