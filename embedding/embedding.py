import re
from importlib.metadata import version
import tiktoken

# Extract our text from the text file
with open("text/the-verdict.txt", "r", encoding="utf-8") as f:
    text = f.read()

preprocess = re.split(r'([,.:;?_!"()\']|--|\s)', text)
preprocess = [item.strip() for item in preprocess if item.strip()]

# Create a vocabularoy so we can tokenize words
all_tokens = sorted(list(set(preprocess)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}

# Simple tokenizer class
class tokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encoder(self, text):
        # Split it into words taking weird symbols into consideration and then strip all the blank spaces from the result
        preprocess = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocess = [item.strip() for item in preprocess if item.strip()]
        preprocess = [item if item in self.str_to_int
                      else "<|unk|>" for item in preprocess]
        ids = [self.str_to_int[s] for s in preprocess]
        return ids
    
    def decoder(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# Imported Bit Par Tokenizer TODO: create own?
tokenizer = tiktoken.get_encoding("gpt2")
text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)