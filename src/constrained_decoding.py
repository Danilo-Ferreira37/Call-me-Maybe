from src.parsing import FuncDef, FuncCall
import json


class ConstrainedDecoding():
    def __init__(s, llm, definition: list[FuncDef], prompts: list[FuncCall]) -> None:
        s.llm = llm
        s.prompts = prompts
        s.file_def = str(definition)
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

                        first_aspas = True
                        input_ids.append(aspas)
                        while input_ids[-1] != aspas or first_aspas:
                            context_token = s.llm.encode(context).squeeze().tolist()
                            context_logits = s.llm.get_logits_from_input_ids(context_token)
                            context_logits[516] = -float('inf')
                            context_logits[14913] = -float('inf')
                            context_logits[6] = -float('inf')
                            context_logits[364] = -float('inf')
                            context_logits[497] = -float('inf')
                            context_logits[3263] = -float('inf')
                            print(s.llm.decode(input_ids))
                            next_tk = context_logits.index(max(context_logits))
                            input_ids.append(next_tk)
                            context += s.llm.decode(next_tk)
                            first_aspas = False

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
                                elif number[-1] == ".":
                                    number += "0"
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
                i = 0
                f_name = []
                context = s.llm.encode(s.file_def).squeeze().tolist()
                context.extend(s.original_input)
                print(s.llm.decode(context))
                current_token = [tk[i] for tk in s.possible_names]
                while (len(current_token)) > 0:
                    s.logits = s.llm.get_logits_from_input_ids(input_ids)
                    restricted_list =  s.restricted_tokens([current_token])
                    



                
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