"""Small pedagogical word-level BPE trainer. Not GPT-style byte-level BPE.

This implementation supports word boundaries and learned adjacent symbol merges;
it is not Unicode-byte BPE and does not reproduce a production tokenizer.
"""
from __future__ import annotations
from collections import Counter
import re


class WordBPETokenizer:
    """Greedy word-BPE: naive training ~O(merges * corpus_symbols) each round.

    Vocabulary construction is repeated for clarity; encode applies O(M*L^2)
    symbol merging for M merges and max symbolized word length L (upper bound).
    """
    def __init__(self, merges: int = 30):
        if merges < 0:
            raise ValueError("merges must be nonnegative")
        self.n_merges = merges
        self.rules: list[tuple[str,str]] = []

    @staticmethod
    def _words(text: str) -> list[str]:
        return re.findall(r"\S+", text.lower())

    @staticmethod
    def _merge(symbols: tuple[str,...], a: str, b: str) -> tuple[str,...]:
        out=[];i=0
        while i<len(symbols):
            if i+1<len(symbols) and symbols[i]==a and symbols[i+1]==b:
                out.append(a+b);i+=2
            else:
                out.append(symbols[i]);i+=1
        return tuple(out)

    def fit(self, corpus: str) -> "WordBPETokenizer":
        vocab=Counter(tuple(list(w)+['</w>']) for w in self._words(corpus))
        self.rules=[]
        for _ in range(self.n_merges):
            pair_counts=Counter()
            for symbols,freq in vocab.items():
                for pair in zip(symbols,symbols[1:]):
                    pair_counts[pair]+=freq
            if not pair_counts:
                break
            # Deterministic on ties, independent of Counter insertion ordering.
            pair=max(pair_counts, key=lambda p:(pair_counts[p],p))
            self.rules.append(pair)
            merged=Counter()
            for symbols,freq in vocab.items():
                merged[self._merge(symbols,*pair)]+=freq
            vocab=merged
        return self

    def encode_word(self, word: str) -> list[str]:
        symbols=tuple(list(word.lower())+['</w>'])
        for pair in self.rules:
            symbols=self._merge(symbols,*pair)
        return list(symbols)

    def encode(self, text: str) -> list[list[str]]:
        return [self.encode_word(word) for word in self._words(text)]
