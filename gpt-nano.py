from typing import Dict

stoi: Dict = lambda chars: { ch: i for i, ch in enumerate(chars)}
itos: Dict = lambda ints: { i: ch for i, ch in enumerate(ints)}
