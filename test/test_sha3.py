# test_sha3.py

from tinygrad.sha3 import SHA3, parallel_sha3_hash

# Test single SHA-3 hashing
data = "Hello, Tinygrad!"
sha3_256 = SHA3(256)
single_hash = sha3_256.hash(data)
print("Single SHA3-256 Hash:", single_hash.hex())

# Test parallel SHA-3 hashing
data_list = ["data1", "data2", "data3"]
parallel_hashes = parallel_sha3_hash(data_list, bit_rate=256)
print("Parallel SHA3-256 Hashes:")
for i, h in enumerate(parallel_hashes):
    print(f"Data {i + 1}: {h.hex()}")
