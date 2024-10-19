from tinygrad import Tensor, nn, dtypes
from tinygrad import Device
from typing import Optional


class Sponge:
    def __init__(self, output_sz: int = 256, rate: Optional[int] = None, capacity: Optional[int] = None):
        self.b = 1600
        self.out_len = output_sz
        if rate is None:
            self.c = output_sz * 2
            self.r = self.b - self.c
        else:
            self.c = capacity
            self.r = rate

        if self.r + self.c != self.b:
            raise ValueError("Rate + Capacity != 1600")
        if self.rc != self.out_len * 2:
            raise ValueError("Capacity must equal 2 * Output Len.")

        self.message = ""

    def to_binary(self, message: str):
        pass
