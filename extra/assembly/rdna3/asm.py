from tinygrad import nn, Tensor

if __name__ == "__main__":
  fns = nn.state.zip_extract(Tensor.from_url("https://gpuopen.com/download/machine-readable-isa/latest/"))
  print(fns)

