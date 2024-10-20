from tinygrad import Tensor, dtypes
from typing import Optional, List
from math import log2
'''
WORK IN PROGRESS
'''


class Sponge:
    def __init__(self, output_sz: int = 256, rate: Optional[int] = None, capacity: Optional[int] = None):
        # Permutation length
        self.b = 1600

        # Lane length
        self.w = int(self.b / 25)

        if self.b % self.w != 0:
            raise ValueError(
                f"Permutation width ({self.b}) must divide by lane width ({self.w})!")

        # Number of rounds in permutation fn
        self.rounds: int = int(12 + (2*log2(self.w)))

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

        message_tnsr: Tensor = Tensor(
            int_string, dtype='uint').unsqueeze(0).detach()

        em = Tensor.empty(0, message_tnsr.shape[1], dtype='uint')

        # Messy but more efficient than looping
        em = em.cat(message_tnsr.rshift(7)).cat(
            message_tnsr.rshift(6)).cat(message_tnsr.rshift(5)).cat(message_tnsr.rshift(4)).cat(message_tnsr.rshift(3))\
            .cat(message_tnsr.rshift(2)).cat(message_tnsr.rshift(1)).cat(message_tnsr.rshift(0)).bitwise_and(1)

        return em.T

    # To-Do; May not need to be separate fn
    def pad(self, data):
        pass

    # Permutation round
    def round(self, lanes: Tensor):
        # Theta step

        # C[x] = A[x,0] xor A[x,1] xor A[x,2] xor A[x,3] xor A[x,4],   for x in 0…4
        # (a + b + c) % 2 = a XOR b XOR c
        sum_x = lanes.sum(axis=1)
        quotient_x = sum_x / 2
        C = sum_x - 2*quotient_x.trunc().cast(dtypes.uint)

        print(C.shape)

    # Permutation Function
    # Implementing Keccak-f1600 to start; generalizing later
    def keccak_fn(self, state: Tensor):
        # lanes = Tensor.zeros((5, 5, self.w), dtype='uint')
        lanes = state.reshape((5, 5, self.w))
        A = self.round(lanes)
        return A
        # try:
        #     lanes = state.reshape((5, 5, self.w))
        #     A = round(lanes)

        # except:
        #     print("ruh roh raggy")

        # for i in range(0, self.rounds):
        #     A = round(lanes)

    # First part of Sponge fns

    def absorb(self, data: Tensor, suff: int = 0x06) -> Tensor:
        state: Tensor = Tensor.zeros(
            self.b, dtype="uint").contiguous()
        suffix: Tensor = Tensor([int(x)
                                for x in bin(suff)[2:]], dtype="uint")

        offset = 0
        while offset < data.numel():
            blockSize: int = min(data.numel() - offset, self.r)
            state[:blockSize] = state[:blockSize].xor(data[offset:offset +
                                                           blockSize])
            offset = offset + blockSize
            if blockSize == self.r:  # i.e. complete block
                # state = Keccak
                blockSize = 0

        # Next step: Handle incomplete blocks

        # print(blockSize+suffix.numel())

        state[blockSize:blockSize+suffix.numel()] = state[blockSize:blockSize +
                                                          suffix.numel()].xor(suffix)
        k = self.keccak_fn(state)
        return state

    def squeeze(self):

        pass

    def digest(self):
        pass


def rot_left_tensor(a, n):
    mask = (1 << 64) - 1
    x = a.lshift(n).bitwise_and(mask).bitwise_or(a.rshift(63))
    print(x.tolist())
    q = x / 2
    r = x - 2*q.cast(dtypes.uint)
    return r
# def rot_left_tensor(tensor, shift):
#     # Assuming the last dimension is the 64-bit word
#     mask = (1 << 64) - 1  # To mask the 64-bit value and prevent overflow
#     shifted: Tensor = ((tensor << shift) & mask) | (tensor >> (63))
#     q: Tensor = shifted / 2
#     rotated = shifted - 2*q.trunc().cast(dtypes.uint)
#     return rotated


def main():
    sp = Sponge()
    message = "If you ever get annoyed, look at me I'm self-employed I love to work at nothing all day\
            And I'll be taking care of business (every day)\
            Taking care of business (every way)\
            I've been taking care of business (it's all mine)\
            Taking care of business and working overtime, work out"

    bt = sp.to_binary(message)
    print(bt[1].numpy())
    print(bin(ord('f')))


if __name__ == "__main__":
    main()
