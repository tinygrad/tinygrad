# Instructions

Set `ARCH` variable to `sm_89` inside the makefile.

Solving the ISA require two sets of input. First is the `.so` file from standard library. Run `make find_libcublasLt`
to make sure it can be found, and the path prefix matches what's in `stdlibs/%.so`. 

Then run `make libcublasLt.txt` to get the solution from the library.

Second input is the generated cuda files by running tinygrad tests because the standard library does not seem to cover
some common operations (specifically, unified memory load and store). I added a block in `cstyle.py`'s render method
to dump the output.

Run `make tinygrad_gen/test_ops` to dump all the kernels. Then run `make tinygrad_gen/test_ops.txt` to use those
kernels to form a second set of solutions. This could take a long time, but most instructions are duplicate. So maybe
you can use a small subset of test_ops to get the same effect.

You should now have `libcublasLt.txt` and `test_ops.txt` in your current directory.
Run `python merge.py sm_89.txt libcublasLt.txt test_ops.txt` to produce the merged solution.

Afterwards, place the file in `assembler/InsAsmRepos` as `DefaultInsAsmRepos.sm_80.txt`.

