import re
from collections import Counter


class BPE:

    PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}{1,3}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self):
        self.byte_encoder = self.byte_encoder()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.vocab = {}          # id -> bytes
        self.inv_vocab = {}      # bytes -> id
        self.merges = {}         # (bytes, bytes) -> rank
        self.pattern = re.compile(self.PATTERN)
        self.special_tokens = {} # str -> id

    def byte_encoder(self) -> dict:
        """
        Build the fixed, reversible byte(0-255) -> visible unicode char map.
        Printable bytes map to themselves; the rest get shifted into an
        unused unicode range so every byte has a printable representative.
        """
        raise NotImplementedError

    def regex_split(self, text: str) -> list[str]:
        """Split raw unicode text into chunks using self.pattern."""
        return self.pattern.findall(text)

    def utf8_encode_chunk(self, chunk: str) -> list[str]:
        """
        Convert one pretokenized chunk to its UTF-8 bytes, then remap each
        byte to its visible-unicode representative via self.byte_encoder.
        Returns a list of single-symbol strings (the BPE base units).
        """
        raw_bytes = chunk.encode("utf-8")
        return [self.byte_encoder[b] for b in raw_bytes]

    def get_pair_counts(self, chunks: list[list[str]]) -> Counter:
        """Count adjacent symbol pairs across all chunks."""
        raise NotImplementedError

    def merge_pair(self, chunks: list[list[str]], pair: tuple, new_symbol: str) -> list[list[str]]:
        """Replace every occurrence of `pair` with `new_symbol` in every chunk."""
        raise NotImplementedError

    def train(self, corpus: list[str], vocab_size: int):
        """
        Full training loop:
          1. regex_split + utf8_encode_chunk every document -> list of chunks
          2. iteratively count pairs, merge the most frequent, record rank
          3. stop once len(vocab) == vocab_size
          4. populate self.vocab / self.inv_vocab / self.merges
        """
        raise NotImplementedError

    def apply_merges(self, symbols: list[str]) -> list[str]:
        """
        Given one chunk's list of symbols, repeatedly apply the
        lowest-rank applicable merge from self.merges until none apply.
        """
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """text -> regex_split -> utf8_encode_chunk -> apply_merges -> ids"""
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """ids -> bytes (via vocab + byte_decoder) -> utf-8 decode -> text"""
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError
    

with open ("text/the-verdict.txt", "r") as f:
    text = f.read()

print(text)