from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, model_name="klue/bert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size
        
    def tokenize(self, text):
        return self.tokenizer(text, padding='max_length', max_length=77, truncation=True, return_tensors='pt')['input_ids']