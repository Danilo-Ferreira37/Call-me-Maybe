from src.parsing import FuncDef, FuncCall


class ConstrainedDecoding():
    def __init__(s, llm, definition: list[FuncDef], prompt: list[FuncCall]):
        s.llm = llm
        s.set_possible_tokens(definition)
        s.prompt = prompt[0].prompt
        s.tokenizer = llm.encode(prompt[0].prompt)
        s.original_input = s.tokenizer.squeeze().tolist()
        s.flow_decoding()

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

    def restricted_tokens(s, token: int) -> list[int]:
        restricted_list = [-float('inf')] * len(s.logits)
        restricted_list[token] = s.logits[token]
        return restricted_list

    def flow_decoding(s):
        input_ids = s.tokenizer.squeeze().tolist()

        # Tokens ID in JSON vocabulary
        open_parent = 58
        open_chav = 90
        close_chav = 92
        description = 4684
        space = 220 
        comma = 11 # virgula
        colon = 25 # dois pontos
        prompt = 40581
        aspas = 1
        name = 606
        parameters = 13786

        count = 0
        wrote_prompt = False
        max_novos_tokens = 80
        wrote_name = False
        idx_func_name = 0

        while count < max_novos_tokens:
            s.logits = s.llm.get_logits_from_input_ids(input_ids)

            if count == 0:
                restricted_list = s.restricted_tokens(open_parent)
                
            elif count == 1:
                restricted_list = s.restricted_tokens(open_chav)

            elif wrote_name:
                for k, v in s.parameters[idx].items():
                    variable = s.llm.encode(k).squeeze().tolist()
                    type = v
                    llm_context_type = s.llm.encode(f"my name ").squeeze().tolist()
                    logit_var_type = s.llm.get_logits_from_input_ids(llm_context_type)

                    input_ids.append(aspas)
                    if isinstance(variable, int):
                        input_ids.append(variable)
                        input_ids.extend([aspas, colon, space])
                    else:
                        input_ids.extend(variable)
                        input_ids.extend([aspas, colon, space])
                    input_ids.append(logit_var_type.index(max(logit_var_type)))
                    input_ids.extend([comma, space])

                break

            # If the last token_id is ('{', 'prompt') token= "
            elif input_ids[-1] in {open_chav, prompt}:
                restricted_list = s.restricted_tokens(aspas)
            elif input_ids[-2] == open_chav and input_ids[-1] == aspas and not wrote_prompt:
                wrote_prompt = True
                restricted_list = s.restricted_tokens(prompt)
            elif input_ids[-2] in [name, prompt, parameters] and input_ids[-1] == aspas:
                restricted_list = s.restricted_tokens(colon)
            elif input_ids[-1] == colon:
                  restricted_list = s.restricted_tokens(space)
            elif input_ids[-4] in [prompt, name]:
                restricted_list = s.restricted_tokens(aspas)

            elif input_ids[-1] == aspas and input_ids[-5] == prompt:
                input_ids.extend(s.original_input)
                input_ids.extend([aspas, comma, space, aspas, name, aspas, colon, space, aspas])
                count += len(s.original_input) + 9
                continue
            elif input_ids[-5] == name:
                context_input = []
                context_input.extend(s.llm.encode(f"{s.prompt} Given the most appropriate function to call is the one that ").squeeze().tolist())
                score = 0
                score_lst = []
                for descpt in s.descriptions:
                    input_description = context_input.copy()
                    for tk_id in descpt[:-1]:
                        logit_description = s.llm.get_logits_from_input_ids(input_description)
                        score += logit_description[tk_id]
                        input_description.append(tk_id)
                    score_lst.append(score)
                    score = 0

                idx_func_name = score_lst.index(max(score_lst))
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
        print(s.llm.decode(input_ids))


#print(s.llm.decode(s.possible_names))
#print(s.possible_names)