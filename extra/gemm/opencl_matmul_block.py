# inspired by:
# https://en.wikipedia.org/wiki/Block_matrix
# https://wrigstad.com/upmarc-ss14/simon3.pdf
# https://cnugteren.github.io/tutorial/pages/page4.html
#
# The current implementation differs the above tutorials/designs:
# 1. read patterns by work item are more "sequential"
# 2. there is no bank conflict resolution
# 3. kernels also support non-square matrices (dim sizes M, K, N can differ)
# 3. kernel 2 and 3 transposes matrix A instead of B
# 4. kernel 3 uses padding to fit all blocks
# Find usage and more details below

import math
import os
import sys

COMP_NUMPY = os.getenv("COMP_NUMPY", 0)
if COMP_NUMPY:
  import numpy

COMP_TORCH = os.getenv("COMP_TORCH", 0)
if COMP_TORCH:
  import torch

from numpy import ndarray
lib_dir = os.path.dirname(__file__) + '\\..\\..\\..\\'
sys.path.append(lib_dir + 'tinygrad')
import tinygrad

os.environ["GPU"] = "1"
import time
import numpy as np

from tinygrad.helpers import getenv, flat_mv
from tinygrad.runtime.ops_gpu import CLAllocator, GPUDevice, CLProgram, CLCompiler

def print_usage():
  print("""
  Usage:
  python opencl_matmul_block.py
  
  Descr:
  Block matrix maltiplication using OpenCL
  
  Options:
    --help: prints this usage
  
  Env vars:
    DTYPE=<float|half|int> The type of data this is equivalent to float32, half16, int32 
    
    M=<n> matrix a height 
    K=<n> matrix a width, matrix b height 
    N=<n> matrix b width
    
    Block params (optional overrides):
    BM =<n> block BA height
    BK =<n> block BA width & block BB height
    BN =<n> block BB width
    BCM=<n> block BC height
    BCN=<n> block BC width
    
    Misc:
    TRIALS=<n> trials to run - Default is 3
    RAND=<0|1> 1 to randomize data otherwise incremental - Default is 1
    COMP=<0|1> validate results - Default is 0
    COMP_NUMPY=<0|1> benchmark against numpy - Default is 0
    COMP_TORCH=<0|1> benchmark against torch - Default is 0
  
  Example cases:
    small matrices:
    M=321 K=953 N=614 python opencl_matmul_block.py
    
    thin K matrices:
    M=2051 N=1011 K=223 python opencl_matmul_block.py
    
    thin M,N matrices:
    M=305 N=210 K=2223 python opencl_matmul_block.py
    
    large matrices:
    M=6121 K=1953 N=1614 python opencl_matmul_block.py
    
    square matrices:
    N=251 python opencl_matmul_block.py
    N=2001 python opencl_matmul_block.py
    
    block-even (blocks fit evenly) matrices:
    M=512 K=256 N=1024 python opencl_matmul_block.py
    M=1024 K=512 N=2048 python opencl_matmul_block.py
    
    square block-even matrices:
    N=256 python opencl_matmul_block.py
    
    large square block-even matrices:
    N=6144 python opencl_matmul_block.py
    
    you can try tuning performance by overriding block params for different data types:
    BM=256 BN=256 BK=8 BCM=16 BCN=16 DTYPE=half N=9000 python opencl_matmul_block.py
  
  """)

if "--help" in sys.argv:
  print_usage()
  exit(0)

def tinygrad_prog(na, nb):
  st = time.perf_counter()
  a = tinygrad.Tensor(na)
  b = tinygrad.Tensor(nb)
  (a @ b).numpy()
  return time.perf_counter() - st

def numpy_prog(na, nb):
  st = time.perf_counter()
  numpy.matmul(na, nb)
  return time.perf_counter() - st

def torch_prog(na, nb):
  st = time.perf_counter()
  a = torch.Tensor(na)
  b = torch.Tensor(nb)
  torch.matmul(a, b)
  return time.perf_counter() - st

def timeit(fxn):
  st = time.perf_counter()
  et = fxn()
  return time.perf_counter() - st

def print_mat(m: ndarray):
  for i in range(0, m.shape[0]):
    s = ""
    for j in range(0, m.shape[1]):
      s += str(m[i][j]) + " "
    print(s)
  print()

def mat_mult(a: ndarray, b: ndarray):
  c = numpy.zeros((a.shape[0], b.shape[1]), dtype=dtype)
  for k in range(a.shape[1]):
    for i in range(a.shape[0]):
      for j in range(b.shape[1]):
        c[i][j] += a[i][k] * b[k][j]
  return c

def assert_equal(a: ndarray, b: ndarray):
  assert a.shape[0] == b.shape[0] and a.shape[1] == b.shape[1]
  for i in range(0, a.shape[0]):
    for j in range(0, a.shape[1]):
      assert a[i][j] == b[i][j], f"a[{i}][{j}] = {a[i][j]} vs b[{i}][{j}] = {b[i][j]}"

def gen_array(M:int, N:int):
  arr = None
  if RAND:
    if dtype==np.half:
      arr = np.zeros(shape=(M, N), dtype=dtype)
      for i in range(0, M):
        arr[i] = np.random.normal(loc=0, scale=1, size=N)
    elif dtype==np.int32:
      arr = np.random.randint(0, 10, (M,N), dtype=dtype)
    else:
      arr = np.random.default_rng().standard_normal(size=(M,N), dtype=dtype)
  else:
    num = 0
    arr = np.zeros(shape=(M,N), dtype=dtype)
    for i in range(0, M):
      for j in range(0, N):
        arr[i][j] = num
        num+=1
  return arr

# TODO: automate finding optimal values
#  Figuring out the block params depends on:
#   a. The size of the input matrices M,K,N
#   b. the amount of HW shared mem
#   c. The kernel max work group size.
#   To choose proper values:
#   1. Query CL_KERNEL_WORK_GROUP_SIZE, this can be tricky because you need to pass a kernel
#     before you create a kernel. Fortunately it seems that you derive the max work group per dimension regardless
#     of the values for BM,BN,BK. In any case expect this will take additional runtime.
#   2. You also need to get the total shared memory for the device see CL_DEVICE_LOCAL_MEM_SIZE.
#     Some reasons to keep these values low is a. to avoid error CL_OUT_OF_RESOURCES, b. the reduced overhead,
#     and c. mem conservation especially if you decide to implement batching in the future.
#   3. Factor in these constraints (See validate_block_params below):
#     a. BM/BCM <= max_local_size, BN/BCN <= max_local_size.
#     b. BK*BM + BK*BN <= total shared memory. (might need to factor in all compute units)
#     c. (BN * BK) % ((BM * BN) / (BCM * BCN)) != 0. see validate_block_params
def get_optimal_block_params(M, K, N) -> (int, int, int, int, int):
  '''
  returns optimal block params: BM, BN, BK, BCM, BCN
  '''

  # for now these default values seems to yield fast results
  if M <= 16 and N <= 16 and K <= 16:
    return 8, 8, 8, 4, 4
  elif M <= 128 and N <= 256 and K <= 256:
    return 16, 16, 8, 4, 4
  elif M <= 512 and N <= 512 and K <= 512:
    return 64, 64, 8, 4, 4
  else:
    return 128, 128, 16, 8, 8
  # also these may be fast for large matrices with DTYPE=half for high end devices
  # 256, 256, 4, 16, 16
  # 256, 256, 8, 16, 16


# block params validation
def validate_block_params():
  if BM&(BM-1) != 0 or BK&(BK-1) != 0 or BN&(BN-1) != 0 or BCM&(BCM-1) != 0 or BCN&(BCN-1) != 0:
    print("Error: BM,BN,BK,BCM,BCN should be powers of two")
    exit(1)

  if (BM * BK) % ((BM * BN) / (BCM * BCN)) != 0:
    print("Error: BM * BK should be a multiple of the number of work items per group: (BM * BN) / (BCM * BCN) (number of work items)\r\n")
    print(f"{BM} * {BK} = {BM * BK}, {BM} * {BN} / ({BCM} * {BCN}) = {BM * BN / (BCM * BCN)}")
    exit(1)

  if (BN * BK) % ((BM * BN) / (BCM * BCN)) != 0:
    print("Error: BN * BK should be a multiple of the number of work items per group: (BM * BN) / (BCM * BCN) (number of work items)\r\n")
    print(f"{BN} * {BK} = {BN * BK}, {BM} * {BN} / ({BCM} * {BCN}) = {BM * BN / (BCM * BCN)}")
    exit(1)

  # TODO: get max local size from CL_KERNEL_WORK_GROUP_SIZE
  max_local_size = 16 # assume this works for most devices
  if BM // BCM > max_local_size:
    print(f"Error: BM / BCM = {BM // BCM} should not exceed max local size {max_local_size}")
    exit(1)
  if BN // BCM > max_local_size:
    print(f"Error: BN / BCM = {BN // BCM} should not exceed max local size {max_local_size}")
    exit(1)

# main
TRIALS = getenv("TRIALS", 3)
COMP = getenv("COMP", 0) # validate results
RAND = getenv("RAND", 1)

# matrices sizes
N = getenv("N", 128)
M = getenv("M", N)
K = getenv("K", N)

# tile sizes (evenly divided by mat sizes)
(BM, BN, BK, BCM, BCN) = get_optimal_block_params(M, K, N)

# override block params
BM = getenv("BM", BM)
BN = getenv("BN", BN)
BK = getenv("BK", BK)
BCM = getenv("BCM", BCM)
BCN = getenv("BCN", BCN)

validate_block_params()

# data type
DTYPE = getenv("DTYPE", "float")
dtype = None
KERNEL_EXT = ""
if DTYPE == "float":
  dtype = np.float32
elif DTYPE == "half":
  dtype = np.half
  KERNEL_EXT = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable"
elif DTYPE == "int":
  dtype = np.int32

device = GPUDevice("GPU")
clalloc = CLAllocator(device)

# input
na = gen_array(M, K)
nb = gen_array(K, N)

FLOPS = M*N*(2*K-1)

# validator
comp = None
if COMP:
  comp = na @ nb # numpy matmul is very off our results since we use very high float (incr)
  # comp = torch.matmul(torch.Tensor(na), torch.Tensor(nb)) # torch
  # comp = mat_mult(na, nb)  # custom validator slower but conforming to c float math

# KERNEL 1
# mat mult using blocks (tiles) and shared memory
# this has provides fast read patterns even with non-square matrices
# since it supports matrices than cannot evenly into blocks it suffers from branch divergence (See if statements)
# Descr:
# for each pair of blocks:
#   all work items are utilized loading portions of data from a,b into blocks BA, BB in shared mem
#   once all loading finishes threads start mat mult of their portion of blocks BA @ BB aggregating into block BC
#   once all the calculations finish block BC is copied over to output memory buffer c
prog_matmult_block = CLProgram(device, "matmult_block", CLCompiler(device, "matmult_block").compile(f"""
#define BM {BM}
#define BN {BN}
#define BK {BK}
#define BCM {BCM}
#define BCN {BCN}

#define dtype {DTYPE}
{KERNEL_EXT}

// tiling
// matrix a needs to be in row major format (M*K)
// matrix b needs to be in row major format (K*N)
// matrix c will be in row major format (M*N)
// block BA will be transposed in col major format (BK*BM)
// block BB will be in row major format (BK*BN)
// block BC will be in row major format (BM*BN)
__kernel void matmult_block(
					const __global dtype* a,
					const __global dtype* b,
					__global dtype* c,
					int M, int K, int N, int WEBA, int WEBB) {{

    const int lclId0 = get_local_id(0);
    const int lclId1 = get_local_id(1);
      
	// offset
    const int offsetm = BM*get_group_id(0);
    const int offsetn = BN*get_group_id(1);
    const int tiles = ceil(K/(float)BK);
	
	// work item for the current work group
	const int witem = lclId1*get_local_size(1) + lclId0;
	
	// offsets for sub matrices
	const int offsetA = witem*WEBA;	
	const int offsetB = witem*WEBB;
	
	// submatrices
    __local dtype BA[BK][BM];
	__local dtype BB[BK][BN];
	dtype BC[BCM][BCN];
	#pragma unroll
    for (int row=0; row<BCM; row++) {{
        #pragma unroll
        for (int col=0; col<BCN; col++) {{
            BC[row][col] = 0.0f;
        }}
    }}
	
    for(int tile=0; tile<tiles; tile++) {{
		
		int offseta = offsetm*K + BK*tile;
		int row = offsetA / BK, col;
		#pragma unroll
		for(int idx=0; idx<WEBA; idx++) {{
			col = (offsetA + idx) % BK;
			if(idx>0 && col == 0) {{
				row++;
			}}
			if(offseta + K*row + col >= K*M)
				break;
			BA[col][row] = a[offseta + K*row + col];
		}}
		
		int offsetb = offsetn + BK*tile*N;
		row = offsetB / BN;
		int offsetbb = offsetb + N*row;
		#pragma unroll
		for(int idx=0; idx<WEBB;idx++) {{
			col = (offsetB + idx) % BN;
			if(idx>0 && col == 0) {{ 
				row++;
				offsetbb = offsetb + N*row;
			}}
			if(offsetbb + col >= K*N) {{
				break;
			}}
			BB[row][col] = b[offsetbb + col];
		}}

        barrier(CLK_LOCAL_MEM_FENCE);

		// partial writes
		const int maxK = K - BK*tile < BK ? K - BK*tile : BK;
		
		for(int ik=0; ik<BK; ik++) {{
			#pragma unroll
			for(int row=0; row<BCM; row++) {{
				#pragma unroll	
				for(int col=0; col<BCN; col++) {{
					if(ik < maxK)
						BC[row][col] += BA[ik][row + BCM*lclId0] * BB[ik][col + BCN*lclId1];
				}}
			}}
		}}

        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    
	
    const int cOffsetRow = offsetm + BCM*lclId0;
	const int cOffsetCol = offsetn + BCN*lclId1;
	
	int idx = cOffsetRow*N + cOffsetCol;
	if(cOffsetCol < N && cOffsetRow < M) {{
		#pragma unroll
		for(int row=0; row<BCM; row++) {{
			#pragma unroll
			for(int col=0; col<BCN; col++) {{
				if((idx + row*N) % N + col >= N)
					continue;
				c[idx + row*N + col] = BC[row][col];
			}}
		}}
	}}
}}
"""))

# KERNEL 2
# mat mult w/ blocks and transpose matrix a
# Just like kernel 1 but matrix a is transposed for faster memory reads
# this requires more memory and processing which reduces the overall benefit
# sometimes yields better results for larger matrices
# suffers from branch divergence just like kernel 1
prog_matmult_block_colmajor = CLProgram(device, "matmult_block_colmajor", CLCompiler(device, "matmult_block_colmajor").compile(f"""
#define BM {BM}
#define BN {BN}
#define BK {BK}
#define BCM {BCM}
#define BCN {BCN}

#define dtype {DTYPE}
{KERNEL_EXT}

// tiling with transposed matrix a for coalesced mem reads
// matrix a needs to be transposed in col major format (K*M)
// matrix b needs to be in row major format (K*N)
// matrix c will be in row major format (M*N)
// block BA will be transposed in col major format (BK*BM)
// block BB will be in row major format (BK*BN)
// block BC will be in row major format (BM*BN)
__kernel void matmult_block_colmajor(
					const __global dtype* a,
					const __global dtype* b,
					__global dtype* c,
					int M, int K, int N, int WEBA, int WEBB) {{

    const int lclId0 = get_local_id(0);
    const int lclId1 = get_local_id(1);
	
	// offset
    const int offsetm = BM*get_group_id(0);
    const int offsetn = BN*get_group_id(1);
    const int tiles = ceil(K/(float)BK);
	
	// work item for the current work group
	const int witem = lclId1*get_local_size(1) + lclId0;
	
	// offsets for sub matrices
	const int offsetA = witem*WEBA;	
	const int offsetB = witem*WEBB;
	
	// submatrices
    __local dtype BA[BK][BM];
	__local dtype BB[BK][BN];
	dtype BC[BCM][BCN];
	#pragma unroll
    for (int row=0; row<BCM; row++) {{
        #pragma unroll
        for (int col=0; col<BCN; col++) {{
            BC[row][col] = 0.0f;
        }}
    }}
	
    for(int tile=0; tile<tiles; tile++) {{
		
		int offseta = offsetm + BK*tile*M;
		int row = offsetA / BM, col;
		#pragma unroll
		for(int idx=0; idx<WEBA; idx++) {{
			col = (offsetA + idx) % BM;
			if(idx>0 && col == 0) {{
				row++;
			}}
			if(offseta + M*row + col >= K*M)
				break;
			BA[row][col] = a[offseta + M*row + col];
		}}
		
		int offsetb = offsetn + BK*tile*N;
		row = offsetB / BN;
		int offsetbb = offsetb + N*row;
		#pragma unroll
		for(int idx=0; idx<WEBB;idx++) {{
			col = (offsetB + idx) % BN;
			if(idx>0 && col == 0) {{ 
				row++;
				offsetbb = offsetb + N*row;
			}}
			if(offsetbb + col >= K*N) {{
				break;
			}}
			BB[row][col] = b[offsetbb + col];
		}}

        barrier(CLK_LOCAL_MEM_FENCE);

		// partial writes
		const int maxK = K - BK*tile < BK ? K - BK*tile : BK;
		
		for(int ik=0; ik<BK; ik++) {{
			#pragma unroll
			for(int row=0; row<BCM; row++) {{
				#pragma unroll	
				for(int col=0; col<BCN; col++) {{
					if(ik < maxK)
						BC[row][col] += BA[ik][row + BCM*lclId0] * BB[ik][col + BCN*lclId1];
				}}
			}}
		}}

        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    
	
    const int cOffsetRow = offsetm + BCM*lclId0;
	const int cOffsetCol = offsetn + BCN*lclId1;
	
	int idx = cOffsetRow*N + cOffsetCol;
	if(cOffsetCol < N && cOffsetRow < M) {{
		#pragma unroll
		for(int row=0; row<BCM; row++) {{
			#pragma unroll
			for(int col=0; col<BCN; col++) {{
				if((idx + row*N) % N + col >= N)
					continue;
				c[idx + row*N + col] = BC[row][col];
			}}
		}}
	}}
}}
"""))

# KERNEL 3
# blocks w/ transpose and padding
# Just like kernel 2 but all matrices are padded so blocks can fit evenly
# this has no branch divergence effects
# this also requires even more memory and processing which reduces the overall benefit
# usually better results for large matrices
# padding is used only if required, so this should be used for matrices that already fit blocks evenly
prog_matmult_block_colmajor_padded = CLProgram(device, "matmult_block_colmajor_padded", CLCompiler(device, "matmult_block_colmajor_padded").compile(f"""
#define BM {BM}
#define BN {BN}
#define BK {BK}
#define BCM {BCM}
#define BCN {BCN}

#define dtype {DTYPE}
{KERNEL_EXT}

// tiling with padded matrices and transposed matrix a for coalesced mem reads
// matrix a needs to be transposed in col major format (K*M)
// matrix b needs to be in row major format (K*N)
// matrix c will be in row major format (M*N)
// block BA will be transposed in col major format (BK*BM)
// block BB will be in row major format (BK*BN)
// block BC will be in row major format (BM*BN)
// Note: all matrices should be padded for dimensions to be multiples of M, N, K
__kernel void matmult_block_colmajor_padded(
					const __global dtype* a,
					const __global dtype* b,
					__global dtype* c,
					const int M, const int K, const int N, int WEBA, int WEBB) {{

    const int lclId0 = get_local_id(0);
    const int lclId1 = get_local_id(1);
			
	// offset
    const int offsetm = BM*get_group_id(0);
    const int offsetn = BN*get_group_id(1);
    const int tiles = K/BK;
	
	// work item for the current work group
	const int witem = lclId1*get_local_size(1) + lclId0;
	
	// offsets for sub matrices
	const int offsetA = witem*WEBA;	
	const int offsetB = witem*WEBB;
	
	// submatrices
    __local dtype BA[BK][BM];
	__local dtype BB[BK][BN];
	dtype BC[BCM][BCN];
	#pragma unroll
    for (int row=0; row<BCM; row++) {{
        #pragma unroll
        for (int col=0; col<BCN; col++) {{
            BC[row][col] = 0.0f;
        }}
    }}
	
    for(int tile=0; tile<tiles; tile++) {{
		
		int offseta = offsetm + BK*tile*M;
		int row = offsetA / BM, col;
		#pragma unroll
		for(int idx=0; idx<WEBA; idx++) {{
			col = (offsetA + idx) % BM;
			if(idx>0 && col == 0) {{
				row++;
			}}
			BA[row][col] = a[offseta + M*row + col];
		}}
		
		int offsetb = offsetn + BK*tile*N;
		row = offsetB / BN;
		int offsetbb = offsetb + N*row;
		#pragma unroll
		for(int idx=0; idx<WEBB;idx++) {{
			col = (offsetB + idx) % BN;
			if(idx>0 && col == 0) {{ 
				row++;
				offsetbb = offsetb + N*row;
			}}
			BB[row][col] = b[offsetbb + col];
		}}

        barrier(CLK_LOCAL_MEM_FENCE);

		for(int ik=0; ik<BK; ik++) {{
			#pragma unroll
			for(int row=0; row<BCM; row++) {{
				#pragma unroll	
				for(int col=0; col<BCN; col++) {{
					BC[row][col] += BA[ik][row + BCM*lclId0] * BB[ik][col + BCN*lclId1];
				}}
			}}
		}}

        barrier(CLK_LOCAL_MEM_FENCE);
    }}

    const int cOffsetRow = offsetm + BCM*lclId0;
	const int cOffsetCol = offsetn + BCN*lclId1;
	
	int idx = cOffsetRow*N + cOffsetCol;
	#pragma unroll
	for(int row=0; row<BCM; row++) {{
		#pragma unroll
		for(int col=0; col<BCN; col++) {{
			if((idx + row*N) % N + col >= N)
				continue;
			c[idx + row*N + col] = BC[row][col];
		}}
	}}
}}
"""))

# transpose
prog_transpose = CLProgram(device, "transpose", CLCompiler(device, "transpose").compile(f"""

#define dtype {DTYPE}
{KERNEL_EXT}

// Simple transpose kernel for a P * Q matrix
__kernel void transpose(const __global dtype* input,
                        __global dtype* output,
                        const int M, const int K,
						const int K2, const int M2) {{
	int glid0 = get_global_id(0);
	int glid1 = get_global_id(1);
	
	if(glid0 >= M || glid1 >= K)
		return;
		
	int idx = glid0 * K + glid1;
	int nIdx = glid1 * M2 + glid0;

	output[nIdx] = input[idx];
}}
"""))

def opencl_prog(na, nb):
  stt = time.perf_counter()

  # output
  nc = np.zeros((M, N), dtype=dtype)

  # used for padding
  m, k, n = M, K, N

  nc_padded = None
  if prog == prog_matmult_block_colmajor_padded:
    paddedm = math.ceil(M / BM) * BM
    paddedk = math.ceil(K / BK) * BK
    paddedn = math.ceil(N / BN) * BN

    if K != paddedk or N != paddedn:
      nb = np.pad(nb, [(0, paddedk - K ), (0, paddedn - N)])
    if M != paddedm or N != paddedn:
      nc_padded = np.zeros((paddedm, paddedn), dtype=dtype)
    m, k, n = paddedm, paddedk, paddedn

  a = clalloc.alloc(M * K * 4)
  b = clalloc.alloc(k * n * 4)
  c = clalloc.alloc(m * n * 4)

  clalloc.copyin(a, memoryview(bytearray(na)))
  clalloc.copyin(b, memoryview(bytearray(nb)))

  # transpose
  at = None
  if prog == prog_matmult_block_colmajor or prog == prog_matmult_block_colmajor_padded:
    at = clalloc.alloc(k * m * 4)
    t_local_size = (16, 16) # assume this is the max local size
    t_global_size = (int(math.ceil(M/t_local_size[0])*t_local_size[0]), int(math.ceil(K/t_local_size[1])*t_local_size[1]))
    # print(f"t_local_size: {t_local_size}, t_global_size: {t_global_size}")

    # convert to tinygrad definition of global size:
    t_global_size = tuple([int(math.ceil(x / y)) for x, y in zip(t_global_size, t_local_size)])

    if prog == prog_matmult_block_colmajor or prog == prog_matmult_block_colmajor_padded:
      prog_transpose(a, at, vals=(M, K, k, m), global_size=t_global_size, local_size=t_local_size, wait=True)

    # DEBUG: view transposed matrix
    # nat = np.zeros((k, m), dtype=dtype)
    # clalloc.copyout(flat_mv(nat.data), at)
    # print("nat", nat)

  # opencl params
  local_size = (BM // BCM, BN // BCN)
  global_size = (int(math.ceil(m / BM)) * BM // BCM, int(math.ceil(n / BN)) * BN // BCN)
  # print(f"local_size: {local_size}, global_size: {global_size}")

  # convert to tinygrad definition of global size
  global_size = tuple([x // y for x, y in zip(global_size, local_size)])

  # calculate total elements that will be loaded by each work item into blocks BA and BB:
  # WEBA = (total elements of block BA) / (total elements of a work group)
  # where: total elements of a work group = (size of output matrix block BC) / (total elements of work item)
  # simplified calculation:
  WEBA = BK * BCM * BCN // BN  # (BM * BK) / ((BM * BN) / (BCM * BCN))
  WEBB = BK * BCM * BCN // BM  # (BK * BN) / ((BM * BN) / (BCM * BCN))

  # also track the time spent only by the kernel
  tmm = None
  if prog == prog_matmult_block_colmajor or prog == prog_matmult_block_colmajor_padded:
    # read the transposed at buffer directly from global mem
    tmm = timeit(lambda: prog(at, b, c, vals=(m, k, n, WEBA, WEBB), global_size=global_size, local_size=local_size, wait=True))
  else:
    tmm = timeit(lambda: prog(a, b, c, vals=(m, k, n, WEBA, WEBB), global_size=global_size, local_size=local_size, wait=True))

  # include copyout into our timings to be fair
  if prog == prog_matmult_block_colmajor_padded and nc_padded is not None:
    # prepare padding only if required
    clalloc.copyout(flat_mv(nc_padded.data), c)
    # copy from padded mat to the final output mat
    nc = nc_padded[0:M, 0:N]
  else:
    clalloc.copyout(flat_mv(nc.data), c)
  ttm = time.perf_counter() - stt

  if COMP:
    if N <= 32:
      print_mat(nc)
      print_mat(comp)
    np.testing.assert_allclose(nc, comp, atol=1e-3)
    # assert_equal(nc, comp)

  clalloc.free(a, m * k * 4)
  clalloc.free(b, n * k * 4)
  clalloc.free(c, m * n * 4)
  if at:
    clalloc.free(at, k * m * 4)

  return ttm, tmm

print(f"TRIALS: {TRIALS}")
print(f"DTYPE: {DTYPE}")
print(f"M: {M}, K: {K}, N: {N}")
print(f"BM: {BM}, BN: {BN}, BK: {BK}, BCM: {BCM}, BCN: {BCN}")
print(f"estimated FLOPs: {FLOPS}")

tms = [tinygrad_prog(na, nb) for _ in range(TRIALS)]
tmtg = min(tms)
print(f"{M*N:10d} {tmtg*1e6:12.2f} us, {tmtg:8.3f} sec(s), would be {FLOPS*1e-9/tmtg:9.2f} total GFLOPS matmul in tinygrad")

if COMP_TORCH:
  tms = [torch_prog(na, nb) for _ in range(TRIALS)]
  tmtr = min(tms)
  print(f"{M*N:10d} {tmtr*1e6:12.2f} us, {tmtr:8.3f} sec(s), would be {FLOPS*1e-9/tmtr:9.2f} total GFLOPS matmul in torch")

if COMP_NUMPY:
  tms = [numpy_prog(na, nb) for _ in range(TRIALS)]
  tmnp = min(tms)
  print(f"{M*N:10d} {tmnp*1e6:12.2f} us, {tmnp:8.3f} sec(s), would be {FLOPS*1e-9/tmnp:9.2f} total GFLOPS matmul in numpy")

prog = prog_matmult_block
tms = [opencl_prog(na, nb) for _ in range(TRIALS)]
tm = min(tms)
print(f"{M * N:10d} {tm[0] * 1e6:12.2f} us, {tm[0]:8.3f} sec(s), would be {FLOPS * 1e-9 / tm[0]:9.2f} total GFLOPS in {prog.name}, "
      f"(mult kernel: {tm[1]:5.3f} sec(s), {FLOPS * 1e-9 / tm[1]:.2f} GFLOPS), "
      f"({(tmtg / tm[0] - 1) * 100:+5.2f}% tinygrad), "
      + (f"({(tmtr / tm[0] - 1) * 100:+5.2f}% torch), " if COMP_TORCH else "")
      + (f"({(tmnp / tm[0] - 1) * 100:+5.2f}% numpy)" if COMP_NUMPY else "")
      )

prog = prog_matmult_block_colmajor
tms = [opencl_prog(na, nb) for _ in range(TRIALS)]
tm = min(tms)
print(f"{M * N:10d} {tm[0] * 1e6:12.2f} us, {tm[0]:8.3f} sec(s), would be {FLOPS * 1e-9 / tm[0]:9.2f} total GFLOPS in {prog.name}, "
      f"(mult kernel: {tm[1]:5.3f} sec(s), {FLOPS * 1e-9 / tm[1]:.2f} GFLOPS), "
      f"({(tmtg / tm[0] - 1) * 100:+5.2f}% tinygrad), "
      + (f"({(tmtr / tm[0] - 1) * 100:+5.2f}% torch), " if COMP_TORCH else "")
      + (f"({(tmnp / tm[0] - 1) * 100:+5.2f}% numpy)" if COMP_NUMPY else "")
      )

prog = prog_matmult_block_colmajor_padded
tms = [opencl_prog(na, nb) for _ in range(TRIALS)]
tm = min(tms)
print(f"{M * N:10d} {tm[0] * 1e6:12.2f} us, {tm[0]:8.3f} sec(s), would be {FLOPS * 1e-9 / tm[0]:9.2f} total GFLOPS in {prog.name}, "
      f"(mult kernel: {tm[1]:5.3f} sec(s), {FLOPS * 1e-9 / tm[1]:.2f} GFLOPS), "
      f"({(tmtg / tm[0] - 1) * 100:+5.2f}% tinygrad), "
      + (f"({(tmtr / tm[0] - 1) * 100:+5.2f}% torch), " if COMP_TORCH else "")
      + (f"({(tmnp / tm[0] - 1) * 100:+5.2f}% numpy)" if COMP_NUMPY else "")
      )