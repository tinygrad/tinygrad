from tinygrad import Tensor, dtypes
from typing import Optional, List

'''
WORK IN PROGRESS
'''


class Sponge:
    def __init__(self, output_sz: int = 256, rate: Optional[int] = None, capacity: Optional[int] = None):
        # Bitrate/State length
        self.b = 1600

        # Size of digest, e.g. SHA256 -> out_len = 256
        self.out_len = output_sz

        # Default
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

    # str -> uint Tensor
    # Not very robust; Will refine once base implementation is complete
    def to_binary(self, message: str) -> Tensor:
        # char -> uint (utf-8)
        int_string: List[int] = [ord(x) for x in message]

        message_tnsr: Tensor = Tensor(int_string, dtype='uint').detach()
        temp_list: List[List[int]] = []

        # uint -> "binary" (still uint)
        for bit in reversed(range(8)):
            temp = message_tnsr.rshift(bit).bitwise_and(1)
            temp_list.append(temp.tolist())

        bit_tensor: Tensor = Tensor(temp_list).permute(1, 0).flatten()

        return bit_tensor

    # To-Do; May not need to be separate fn
    def pad(self, data):
        pass

    # First part of Sponge fns
    def absorb(self, data: Tensor, suffix=None):
        state: Tensor = Tensor.zeros(self.b)
        block_i = 0

        # m_0, m_1,..., m_n
        blocks = data.split(self.r)

        while block_i < blocks.numel():
            if blocks[block_i].numel() == state.numel():  # i.e. complete block
                state = state.xor(blocks[block_i])
            block_i += 1
        # Next step: Handle incomplete blocks

    def squeeze(self):
        pass

    def digest(self):
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


if __name__ == "__main__":
    main()
