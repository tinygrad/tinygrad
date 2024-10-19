from tinygrad import Tensor, nn, dtypes
from tinygrad import Device
from typing import Optional, List
import numpy as np


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
        if self.c != self.out_len * 2:
            raise ValueError("Capacity must equal 2 * Output Len.")

        self.message = ""

    def to_binary(self, message: str) -> Tensor:
        int_string: List[int] = [ord(x) for x in message]
        message_tens: Tensor = Tensor(int_string, dtype='uint').detach()
        temp_list: List[List[int]] = []

        for bit in reversed(range(8)):
            temp = message_tens.rshift(bit).bitwise_and(1)
            temp_list.append(temp.tolist())

        bit_tensor: Tensor = Tensor(temp_list).permute(1, 0).flatten()

        return bit_tensor

    def pad(self, data):
        pass

    def absorb(self, data: Tensor, suffix=None):
        state: Tensor = Tensor.zeros(self.b)
        block_i = 0
        blocks = data.split(self.r)

        while block_i < blocks.numel():
            if blocks[block_i].numel() == state.numel():
                state = state.xor(blocks[block_i])
            block_i += 1
        pass


def main():
    sp = Sponge()
    message = """
            If you ever get annoyed, look at me I'm self-employed
            I love to work at nothing all day
            And I'll be taking care of business (every day)
            Taking care of business (every way)
            I've been taking care of business (it's all mine)
            Taking care of business and working overtime, work out
        """

    bt = sp.to_binary(message)

    print(bt.numpy())

    # print(bt)


main()
