from src.parsing import FuncFormat, PromptItem

class ConstrainedDecoding():
    def __init__(self, llm, config: FuncFormat | PromptItem):
        self.llm = llm
        self.config = config
        self.tokenizer = llm.encode(config[0].prompt)
        self.logits = self.llm.get_logits_from_input_ids(self.tokenizer.tolist()[0])

        
    def flow_decoding(self):
        