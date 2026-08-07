import regex as re
from collections import Counter


class bpe:
    # Or's through the different types
    PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}{1,3}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self):
        self.pattern = re.compile(self.PATTERN)
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}
        self.output = []

    # Use that pattern we made with the or checks
    def regex_split(self, text):
        return self.pattern.findall(text)

    # Take those token we just split up and encode them (utf bytes)
    def utf_encoding(self, tokens):
        return [token.encode("utf-8") for token in tokens]

    # Id our bytes from vocab 
    def byte_to_id(self):
        return {v: k for k, v in self.vocab.items()}

    # Make a list of the ids we have from our utf bytes
    def get_ids(self, encoded_tokens):
        return [list(token) for token in encoded_tokens]

    # Use the learned merges on new text and return its ids
    def encoding(self, text):
        regex = self.regex_split(text)
        encoding = self.utf_encoding(regex)
        id_list = self.get_ids(encoding)

        for pair, new_id in self.merges.items():
            id_list = [self.merge(ids, pair, new_id) for ids in id_list]
        
        output = [self.output + idx for idx in id_list]
        return output




    # Turns ids back into its original text
    def decode(self, ids):
        byte_seq = b"".join(self.vocab[i] for i in ids)
        return byte_seq.decode("utf-8")

    # Merge two ids into one
    def merge(self, ids, pair, new_id):
        merged = []
        i = 0
        while i < len(ids):
            if  i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1

        return merged

    # Count tof often id pair occures 
    def count_pairs(self, ids_list):
        pairs = []
        for inner in ids_list:
            for i in range(len(inner) - 1):
                pairs.append((inner[i], inner[i+1]))

        return Counter(pairs)

    # Return the most frequent byte pair
    def most_frequent_pair(self, pair_counts):
        best = 0
        result = None
        for k, v in pair_counts.items():
            if v > best:
                best = v
                result = k
        
        return result

    # Merge the num_merges most frequent pairs
    def train(self, ids_list, num_merges):

        for i in range(num_merges):
            pair_count = self.count_pairs(ids_list)
            most_frequent = self.most_frequent_pair(pair_count)   

            if most_frequent is None:
                break

            new_id = 256 + i
            ids_list = [self.merge(ids, most_frequent, new_id) for ids in ids_list]
            self.merges[most_frequent] = new_id
            self.vocab[new_id] = self.vocab[most_frequent[0]] + self.vocab[most_frequent[1]]

        return ids_list


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
    


train_text = "the cat sat on the mat. the cat ran fast."

tk = bpe()
print(tk.encoding(train_text))