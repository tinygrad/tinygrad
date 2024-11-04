# sha3.py
# Lightweight, modern SHA-3 implementation for Tinygrad with parallel processing support

from typing import List, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib


@dataclass
class SHA3:
    bit_rate: int = 256

    def __post_init__(self):
        # Use hashlib's SHA3 functions based on bit rate
        self.hash_func = self._select_hashlib_sha3()

    def _select_hashlib_sha3(self):
        if self.bit_rate == 224:
            return hashlib.sha3_224
        elif self.bit_rate == 256:
            return hashlib.sha3_256
        elif self.bit_rate == 384:
            return hashlib.sha3_384
        elif self.bit_rate == 512:
            return hashlib.sha3_512
        else:
            raise ValueError("Unsupported SHA3 bit rate")

    def hash(self, data: Union[bytes, str]) -> bytes:
        if isinstance(data, str):
            data = data.encode('utf-8')
        hasher = self.hash_func()
        hasher.update(data)
        return hasher.digest()


def parallel_sha3_hash(data_list: List[Union[bytes, str]], bit_rate: int = 256) -> List[bytes]:
    """
    Compute SHA-3 hashes for multiple data items in parallel.

    Args:
        data_list (List[Union[bytes, str]]): List of data items to hash.
        bit_rate (int): Bit rate for SHA-3 hash function (224, 256, 384, 512).

    Returns:
        List[bytes]: List of computed hash digests.
    """
    sha3_instance = SHA3(bit_rate)
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(sha3_instance.hash, data_list))
    return results
