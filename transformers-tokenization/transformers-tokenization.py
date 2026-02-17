import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.word_to_id["<PAD>"] = 0
        self.word_to_id["<UNK>"] = 1
        self.word_to_id["<BOS>"] = 2
        self.word_to_id["<EOS>"] = 3
        
        idx = 4
        for text in texts:
            for word in text.split():
                if word not in self.word_to_id:
                     self.word_to_id[word] = idx
                     idx += 1
        
        self.vocab_size = len(self.word_to_id.keys())
        self.id_to_word = {v:k for k, v in self.word_to_id.items()}

    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        
        out = []
        for word in text.split():
            out.append(self.word_to_id.get(word, 1))
        return out

    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        out = []
        if ids:
            for id in ids:
                out.append(self.id_to_word[id])
        return " ".join(out)

