from typing import List
from datasets import load_dataset

d_train = load_dataset("tiny_shakespeare", split="train")[0]["text"]
d_val = load_dataset("tiny_shakespeare", split="validation")[0]["text"]

vocab = sorted(list(set(d_train + d_val)))

stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}


def encode(s: str):
    return [stoi[c] for c in s]


def decode(l: List):
    return "".join([itos[i] for i in l])
