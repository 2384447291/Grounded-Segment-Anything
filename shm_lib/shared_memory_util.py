from typing import Tuple
from dataclasses import dataclass
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from atomics import atomicview, MemoryOrder, UINT

@dataclass
class ArraySpec:
    name: str
    shape: Tuple[int]
    dtype: np.dtype


class SharedAtomicCounter:
    def __init__(self, 
            shm_manager: SharedMemoryManager, 
            size :int=8 # 64bit int
            ):
        shm = shm_manager.SharedMemory(size=size)
        self.shm = shm
        self.size = size
        self.store(0) # initialize

    @property
    def buf(self):
        return self.shm.buf[:self.size]

    def load(self) -> int:
        with atomicview(buffer=self.buf, atype=UINT) as a: 
            value = a.load(order=MemoryOrder.ACQUIRE)
        return value
    
    def store(self, value: int):
        with atomicview(buffer=self.buf, atype=UINT) as a:
            a.store(value, order=MemoryOrder.RELEASE)
    
    def add(self, value: int):
        with atomicview(buffer=self.buf, atype=UINT) as a:
            a.add(value, order=MemoryOrder.ACQ_REL)


def encode_text_prompt(text: str, max_length: int = 256) -> np.ndarray:
    """Encode a string into a fixed-size numpy array."""
    encoded = text.encode('utf-8')
    if len(encoded) > max_length:
        raise ValueError("Prompt is too long.")
    
    buffer = np.zeros(max_length, dtype=np.uint8)
    buffer[:len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return buffer

def decode_text_prompt(encoded_array: np.ndarray) -> str:
    """Decode a numpy array back to a string."""
    null_idx = np.where(encoded_array == 0)[0]
    if len(null_idx) > 0:
        encoded_array = encoded_array[:null_idx[0]]
    return encoded_array.tobytes().decode('utf-8')
