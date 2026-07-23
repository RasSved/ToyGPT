import re

class tokenizer(object):
    def __init__ (self, text):
        preprocessed = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
        preprocessed = preprocessed.split()

        word_list = sorted(set(preprocessed))
        word_list.extend(["<|unk|>", "<|END|>"])

        self.str_to_int = {word:id for id, word in enumerate(word_list)}
        self.int_to_str = {id:word for word, id in self.str_to_int.items()}
        

    def encoder(self, text):
        # Convert raw text into a list of integer IDs
        preprocessed = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
        preprocessed = preprocessed.split()

        unk_id = self.str_to_int["<|unk|>"]
        mapping = [self.str_to_int.get(word, unk_id) for word in preprocessed]
        return mapping

    def decoder(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        return text 


with open("text/the-verdict.txt", "r") as f:
    text = f.read()

print(text)

tokenized = tokenizer(text)


text2 = "testing testing"
test2 = [5, 12, 3]

test = tokenized.decoder(test2)

print(test)
