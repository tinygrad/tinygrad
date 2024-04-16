tinygrad has four pieces

* frontend (Tensor -> LazyBuffer)
  * See tensor.py, function.py, multi.py, and lazy.py
  * The user interacts with the Tensor class
  * This outputs LazyBuffers, which form the simple compute graph
* scheduler (LazyBuffer -> ScheduleItem)
  * See engine/schedule.py
  * When a Tensor is realized, the scheduler is run to get its LazyBuffers to be computed
  * This takes in LazyBuffers and groups them as appropriate into kernels.
  * It returns a list of ScheduleItems + all the Variables used in the graph
* lowering (TODO: lots of work to clean this up still)
  * See codegen/ (ScheduleItem.ast -> UOps)
    * ScheduleItems have an ast that's compiled into actual GPU code
    * Many optimization choices can be made here, this contains a beam search.
  * renderer/compiler (UOps -> machine code)
    * UOps are tinygrad's IR, similar to LLVM IR
    * Here we either convert them to a high level language or machine code directly
  * engine/realize.py (ScheduleItem -> ExecItem)
* runtime
  * See runtime/
  * Runtime actually interacts with the GPUs
  * It manages Buffers, Programs, and Queues
  * Sadly, METAL and GPU (OpenCL) don't have a compiler that can be pulled out from the device itself

