from src.parsing import FuncDef, FuncCall


class ConstrainedDecoding():
    def __init__(s, llm, definition: list[FuncDef], prompt: list[FuncCall]):
        s.llm = llm
        s.set_possible_tokens(definition)
        s.prompt = prompt
        s.tokenizer = llm.encode(prompt[0].prompt)
        s.original_input = s.tokenizer.squeeze().tolist()
        s.flow_decoding()

    def set_possible_tokens(s, definition: list[FuncDef]) -> None:
        possible_names = ""
        for dct in definition:
            possible_names += f"{dct.name}"
        s.possible_names = s.llm.encode(possible_names).squeeze()
        print(s.possible_names)
    def flow_decoding(s):
        input_ids = s.tokenizer.squeeze().tolist()
        # Tokens ID in JSON vocabulary
        open_parent = 58
        open_chav = 90
        close_chav = 92
        space = 220 
        comma = 11 # virgula
        colon = 25 # dois pontos
        prompt = 40581
        aspas = 1
        name = 606
        underline = 62
        fn = 8822
        parameters = 13786

        count = 0
        max_novos_tokens = 50
        while count < max_novos_tokens:
            s.logits = s.llm.get_logits_from_input_ids(input_ids)

            if count == 0:
                restricted_list = [-float('inf')] * len(s.logits)
                restricted_list[open_parent] = s.logits[open_parent]
            elif count == 1:
                restricted_list= [-float('inf')] * len(s.logits)
                restricted_list[open_chav] = s.logits[open_chav]

            # If the last token_id is ('{', 'prompt') token= "
            elif input_ids[-1] in {open_chav, prompt}:
                restricted_list = [-float('inf')] * len(s.logits)
                restricted_list[aspas] = s.logits[aspas]
            elif input_ids[-2] == open_chav and input_ids[-1] == aspas:
                restricted_list = [-float('inf')] * len(s.logits)
                restricted_list[prompt] = s.logits[prompt]
            elif input_ids[-2] in {name, prompt, parameters} and input_ids[-1] == aspas:
                restricted_list = [-float('inf')] * len(s.logits)
                restricted_list[colon] = s.logits[colon]
            elif input_ids[-1] == colon:
                restricted_list = [-float('inf')] * len(s.logits)
                restricted_list[space] = s.logits[space]
            elif input_ids[-4] in {prompt, name}:
                restricted_list = [-float('inf')] * len(s.logits)
                restricted_list[aspas] = s.logits[aspas]
            elif input_ids[-1] == aspas and input_ids[-5] == prompt:
                input_ids.extend(s.original_input)
                input_ids.extend([aspas, comma, space, aspas, name, aspas, colon, space, aspas])
                count += len(s.original_input) + 9
                continue
            elif input_ids[-5] == name:
                restricted_list = [-float('inf')] * len(s.logits)
                for n in s.possible_names:
                    idx = restricted_list.index(n)
                    restricted_list[idx] = s.logits[idx]
            else:
                restricted_list = s.logits
            
            max_logit = max(restricted_list)
            idx = restricted_list.index(max_logit)
            input_ids.append(idx)

            count += 1
            
        print(s.llm.decode(input_ids))