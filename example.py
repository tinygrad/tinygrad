from tinygrad import Tensor, dtypes

x_cpu = Tensor.randn((64 * 1024))
x_ref = x_cpu.exp().tolist()

x = x_cpu.to("tt")
z = x.exp()

# Currently returns the wrong results because the exp kernel works on bfloat16
print(z.tolist()[0:10])
print(x_ref[0:10])

print("Finished")
