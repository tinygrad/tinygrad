from tinygrad import Tensor

"""
From the readme

DEBUG=3 python3 -c "from tinygrad import Tensor;
N = 1024; a, b = Tensor.rand(N, N), Tensor.rand(N, N);
c = (a.reshape(N, 1, N) * b.T.reshape(1, N, N)).sum(axis=2);
print((c.numpy() - (a.numpy() @ b.numpy())).mean())"

simplified (3x3 @ 3x3) example below (from this video: https://www.youtube.com/watch?v=otXuqjoCs2M)

"""

a = Tensor([
    [-2,5,1],
    [0,8,-7],
    [9,-4,-3]
])
print("a:") 
print(f"{a.numpy()}")

b = Tensor([
    [3,-4,6],
    [-5,2,-1],
    [8,9,0]
])
print("b:") 
print(f"{b.numpy()}")


a_reshaped = a.reshape(3, 1, 3)
print("a_reshaped:") 
print(f"{a_reshaped.numpy()}")

b_reshaped = b.T.reshape(1, 3, 3)
print("b_reshaped:") 
print(f"{b_reshaped.numpy()}")

c = (a_reshaped * b_reshaped).sum(axis=2)
print("c:") 
print(f"{c.numpy()}")

c_numpy = a.numpy() @ b.numpy()
print("c_numpy:") 
print(f"{c_numpy}")

"""
c's result
[
    [-23, 27, -17],
    [-96, -47, -8],
    [23, -71, 58]
]
"""

print("c_numpy - c:") 
print(f"{c_numpy - c.numpy()}")


