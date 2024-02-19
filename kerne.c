// task 0: define data pointers
kernel void E_4(device int* data0, device int* data1, device int* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  // task 1: define dimensions
  int gidx0 = gid.x; /* 4 */

  // task 2: do "compute"
  int val0 = *(data1+gidx0);
  int val1 = *(data2+gidx0);

  // task3: store the "compute result"
  *(data0+gidx0) = (val0+val1);
}


// task 0: define data pointers
kernel void r_4(device int* data0, device int* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  // task 1: define dimensions 

  // task 2: do "compute"
  int acc0 = 0;
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    int val0 = *(data1+ridx0);
    acc0 = (val0+acc0);
  }

  // task 3: store "compute"
  *(data0+0) = acc0;
}

// --- pre-"compute"
SPECIAL = auto(); // this is used to get grid/thread ids
DEFINE_GLOBAL = auto(); // define global data share pointers
DEFINE_LOCAL = auto(); // define local data share pointers
DEFINE_ACC = auto(); // define local reduce

// --- control
LOOP = auto(); ENDLOOP = auto(); // render outer part of reduce
BARRIER = auto(); IF = auto(); ENDIF = auto(); // post-sync compute

// --- data movement
LOAD = auto(); GEP = auto(); // scalar, vector load
STORE = auto(); // scalar, vector store (handled based on its dtype)
CONST = auto();

// --- ALU
PHI = auto(); // ReduceOps
ALU = auto(); // other (UnaryOps, BinaryOps, TernaryOps)
WMMA = auto(); // special for TensorCores, tinygrad's opset doesnt have it. it's derived from a matmul kernel
CAST = auto(); // bitcast and normal

// pov 1: there are two types of compute: reduce, linear. reduce ops store PHI, otherwise we store the final UOp
// pov 2: reduce ops are fundmentally a higher level than low-level linear operations. in other words, they can superset linear ops within them?
// -> pov1 is the simplest way to think about compute uops.

// the loop is just expressed at a different abstraction level (global size)

// --- task 0: define data pointers
kernel void r_256_16_16(device int* data0, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  threadgroup int temp[16];

  // task 1: define dimensions
  int gidx0 = gid.x; /* 256 */
  int lidx1 = lid.x; /* 16 */

  // task 2: do "compute"
  int acc0 = 0;
  for (int ridx0 = 0; ridx0 < 16; ridx0++) {
    acc0 = (((((gidx0*(-1))+(lidx1*(-16))+(ridx0*(-1)))<(-254))?1:0)+acc0);
  }

  // task 3; store "compute" result
  *(temp+lidx1) = acc0;

  // task 2: do "compute" (a compute node can optionally have a barrier if its deps are not on-chip)
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((lidx1<1)) {
    int acc1 = 0;
    for (int ridx1 = 0; ridx1 < 16; ridx1++) {
      int val0 = *(temp+ridx1);
      acc1 = (val0+acc1);
    }

    // task 3: store "compute" result
    *(data0+gidx0) = (acc1+(-1));
  }
}
