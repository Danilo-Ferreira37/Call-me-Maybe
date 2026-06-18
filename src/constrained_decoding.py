from src.parsing import FuncFormat, PromptItem


class ConstrainedDecoding():
    def __init__(self, llm, config: FuncFormat | PromptItem):
        self.llm = llm
        self.config = config
        self.tokenizer = llm.encode(config[0].prompt)

        self.flow_decoding()

    def exclude_impossible_tokens() -> None:
        pass


    def flow_decoding(self):
        input_ids = self.tokenizer.squeeze().tolist()

        open_parent = 677
        open_chav = 90
        
        count = 0
        max_novos_tokens = 50
        
        while count < max_novos_tokens:
            self.logits = self.llm.get_logits_from_input_ids(input_ids)
            
            if count == 0:
                restricted_list = [-float('inf')] * len(self.logits)
                restricted_list[open_parent] = self.logits[open_parent]
                
            elif count == 1:
                restricted_list= [-float('inf')] * len(self.logits)
                restricted_list[open_chav] = self.logits[open_chav]


            else:
                restricted_list = self.logits
            
            max_logit = max(restricted_list)
            idx = restricted_list.index(max_logit)
            
            input_ids.append(idx)
            count += 1
            
        print(self.llm.decode(input_ids))