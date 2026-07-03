from src.parsing import FuncDef, FuncCall
import json


class ConstrainedDecoding():
    def __init__(s, llm, definition: list[FuncDef], prompts: list[FuncCall]) -> None:
        s.llm = llm
        s.prompts = prompts
        s.context_file = json.dumps(
                                    [d.model_dump() for d in definition],
                                    ensure_ascii=False,
                                    indent=2)
        s.set_possible_tokens(definition)
        s.output = "["
        s.output_manager()
        print(s.output)
        data = json.loads(s.output)
        with open("data/output.json", "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        

    def set_possible_tokens(s, definition: list[FuncDef]) -> None:
        s.possible_names = []
        s.descriptions = []
        s.parameters = []

        for i, dct in enumerate(definition):
            parameter = {}
            s.possible_names.append(s.llm.encode(dct.name).squeeze().tolist())
            s.descriptions.append(s.llm.encode(dct.description).squeeze().tolist())
            for para_name, para_type in definition[i].parameters.items():
                parameter[para_name] = str(para_type).replace("type=", "").replace("'", "")
            s.parameters.append(parameter)

    def restricted_tokens(s, tokens: list[int]) -> list[int]:
        restricted_list = [-float('inf')] * len(s.logits)
        for tk in tokens:
            restricted_list[tk] = s.logits[tk]
        return restricted_list

    def output_manager(s):
        for i in range(len(s.prompts)):
            s.prompt = s.prompts[i].prompt
            s.original_input = s.llm.encode(s.prompt).squeeze().tolist()
            s.flow_decoding()
            if i < len(s.prompts) - 1:
                s.output += ","

        s.output += "]"


    def flow_decoding(s):
        # Tokens ID in JSON vocabulary
        open_chav = 90
        close_chav = 92
        true = 1866
        false = 3849
        space = 220 
        comma = 11 # virgula
        colon = 25 # dois pontos
        prompt = 40581
        point = 13
        aspas = 1
        name = 606
        zero = 15
        parameters = 13786

        input_ids = [open_chav, aspas, prompt, aspas, colon, space]
        count = 0
        max_novos_tokens = 80
        wrote_name = False
        idx_func_name = 0

        while count < max_novos_tokens:
            s.logits = s.llm.get_logits_from_input_ids(input_ids)

            if wrote_name:
                add_context = ""
                para_count = 0
                for k, type_var in s.parameters[idx_func_name].items():
                    para_count += 1 
                    variable = s.llm.encode(k).squeeze().tolist()
                    
                    input_ids.append(aspas)
                    if isinstance(variable, int):
                        input_ids.append(variable)
                        input_ids.extend([aspas, colon, space])
                    else:
                        input_ids.extend(variable)
                        input_ids.extend([aspas, colon, space])

                    if type_var == "string":
                        context = f'Given the question {s.prompt}, the string of parameter "{k}" is "'

                        input_ids.append(aspas)
                        verify = ""
                        while True:
                            context_token = s.llm.encode(context).squeeze().tolist()
                            context_logits = s.llm.get_logits_from_input_ids(context_token)
                            next_tk = context_logits.index(max(context_logits))
                            verify += s.llm.decode(next_tk)
                            if '"' in verify:
                                input_ids.append(aspas)
                                break
                            input_ids.append(next_tk)
                            context += s.llm.decode(next_tk)
                            # tenho que dar decode do parametro string e apagar o que esta apos as aspas e dar encode denovo

                    elif type_var == "boolean":
                        context_logits = s.restricted_tokens([false, true])
                        input_ids.append(context_logits.index(max(context_logits)))
                    
                    elif type_var == "number":
                        context = f"Given the question {s.prompt},{add_context} the value of parameter '{k}' is "
                        number = ""
                        while True:
                            context_token = s.llm.encode(context).squeeze().tolist()
                            context_logits = s.llm.get_logits_from_input_ids(context_token)
                            next_token = context_logits.index(max(context_logits))
                            if (next_token >= 15 and next_token <= 24) or next_token == point:
                                input_ids.append(next_token)
                                context += s.llm.decode(next_token)
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
                        context = f"Given the question {s.prompt},{add_context} the value of parameter '{k}' is "
                        number = ""
                        while True:
                            context_token = s.llm.encode(context).squeeze().tolist()
                            context_logits = s.llm.get_logits_from_input_ids(context_token)
                            next_token = context_logits.index(max(context_logits))
                            if (next_token >= 15 and next_token <= 24):
                                input_ids.append(next_token)
                                context += s.llm.decode(next_token)
                                number += s.llm.decode(next_token)
                            else:
                                break
                        add_context += f"where {k} = {number}, "


                    if para_count == len(s.parameters[idx_func_name]):
                        input_ids.append(close_chav)
                    else:
                        input_ids.extend([comma, space])
                break

            elif input_ids[-4] in [prompt, name]:
                restricted_list = s.restricted_tokens([aspas])

            elif input_ids[-1] == aspas and input_ids[-5] == prompt:
                input_ids.extend(s.original_input)
                input_ids.extend([aspas, comma, space, aspas, name, aspas, colon, space, aspas])
                count += len(s.original_input) + 9
                continue
   
            elif input_ids[-5] == name:
                f_name = []
                context = []
                for i in range(len(s.possible_names)):
                    context.extend(s.possible_names[i])
                    context.extend([colon, space])
                    context.extend(s.descriptions[i])
                    context.append(space)
                context.extend(s.llm.encode("User request: ").squeeze().tolist())
                context.extend(s.original_input)
                context.extend(s.llm.encode(" The most appropriate function to call is: ").squeeze().tolist())

                #print(s.llm.decode(context))
                #exit(0)
                candidate_names = s.possible_names.copy()
                i = 0
                while len(candidate_names) > 1:
                    s.logits = s.llm.get_logits_from_input_ids(context)
                    possibles_tk = [tk[i] for tk in candidate_names]
                    restricted_list =  s.restricted_tokens(possibles_tk)
                    next_tk = restricted_list.index(max(restricted_list))
                    f_name.append(next_tk)
                    context.append(next_tk)
                    for candidate in candidate_names.copy():
                        if next_tk != candidate[i]:
                            candidate_names.remove(candidate)
                    i += 1
                f_name = candidate_names[0]
                idx_func_name = s.possible_names.index(f_name)
                input_ids.extend(s.possible_names[idx_func_name])
                input_ids.extend([aspas, comma, space, aspas, parameters, aspas, colon, space, open_chav])
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
        print(s.llm.decode(input_ids))
        


#print(s.llm.decode(s.possible_names))
#print(s.possible_names)