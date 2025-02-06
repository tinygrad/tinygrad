import time
import numpy
from tinygrad import Tensor



q, k, v = Tensor.rand(1,1,20000,768), Tensor.rand(1,1,20000,768), Tensor.rand(1,1,20000,768)



start = time.time()

res,_ = q.flash_attention(k,v,5000,5000, is_casual=True)
print(res.abs().sum().numpy())

end = time.time()
print(f"flash attention first run time : {end - start}")
print()


start = time.time()

print(q.scaled_dot_product_attention(k,v).abs().sum().numpy())

end = time.time()
print(f"normal attention first run time : {end - start}")
print()



q, k, v = Tensor.rand(1,1,20000,768), Tensor.rand(1,1,20000,768), Tensor.rand(1,1,20000,768)

start = time.time()

res,_ = q.flash_attention(k,v,5000,5000, is_casual=True)
print(res.abs().sum().numpy())

end = time.time()
print(f"flash attention second run time : {end - start}")
print()


start = time.time()

print(q.scaled_dot_product_attention(k,v).abs().sum().numpy())

end = time.time()
print(f"normal attention second run time : {end - start}")
print()




q, k, v = Tensor.rand(1,1,20000,768), Tensor.rand(1,1,20000,768), Tensor.rand(1,1,20000,768)
res,_ = q.flash_attention(k,v,5000,5000, is_casual=True)
print(f"error of the flash attention {100 * (q.scaled_dot_product_attention(k,v) - res).abs().sum().numpy() / res.abs().sum().numpy():.5f}% (casual)")

res,_ = q.flash_attention(k,v,5000,5000, is_casual=False)
mask = Tensor.full((q.shape[0], q.shape[1], q.shape[2], q.shape[2]), float("-inf")).triu(1)
print(f"error of the flash attention {100 * (q.scaled_dot_product_attention(k,v, attn_mask=mask) - res).abs().sum().numpy() / res.abs().sum().numpy():.5f}% (with mask)")
print()

q, k, v = Tensor.rand(1,1,50000,768), Tensor.rand(1,1,50000,768), Tensor.rand(1,1,50000,768)


try:
    print("Trying normal attention on a big sequence")
    q.scaled_dot_product_attention(k,v).numpy()
except Exception as e:
    print(f"failure: {e}")
else:
    print("success")

try:
    print("Trying flash attention on a big sequence")
    q.flash_attention(k,v,5000,5000, is_casual=True)[0].numpy()
except Exception as e:
    print(f"failure: {e}")
else:
    print("success")
