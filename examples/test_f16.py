from tinygrad import Tensor, dtypes

if __name__ == "__main__":
    a = Tensor([1.0,2.0,3.0], dtype=dtypes.half)
    print((a*2.0).numpy())