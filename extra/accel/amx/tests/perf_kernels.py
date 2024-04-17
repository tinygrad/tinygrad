f = open("perf_kernels.c", "w")
f.write('#include <stdint.h>\n')
ops = [10, 12, 14, 15, 18, 19, 20, 21]
ops.sort()
for op in ops:
    for nreg in range(1, 17):
        f.write(f'static void perf_{op}_{nreg}(uint64_t count, const uint64_t* args){{\n')
        for i in range(nreg):
            if i: f.write(" ")
            f.write(f'uint64_t a{i} = args[{i}];')
        unroll_count = 1
        count_shift = 0
        while unroll_count * nreg < 8:
            unroll_count <<= 1
            count_shift += 1
        if count_shift:
            f.write(f' count >>= {count_shift};')
        f.write('\n__asm("1:')
        for j in range(unroll_count):
            for i in range(1, nreg+1):
                f.write(f'.word ({0x201000 + (op << 5)} + 0%{i} - ((0%{i} >> 4) * 6))\\n')
        f.write(f'subs %0, %0, #1\\nbne 1b')
        f.write(f'" : "+r"(count) : ')
        for i in range(nreg):
            if i:
                f.write(", ")
            f.write(f'"r"(a{i})')
        f.write(f' : "memory");\n')
        f.write(f"}}\n")
f.write('typedef void (*perf_kernel_t)(uint64_t, const uint64_t*);\n')
f.write('static perf_kernel_t perf_kernels[] = {\n')
for op in range(ops[0], ops[-1] + 1):
    for nreg in range(1, 17):
        f.write(f'&perf_{op}_{nreg}' if op in ops else '0')
        f.write(', ')
    f.write('\n')
f.write('};\n')
f.write('perf_kernel_t get_perf_kernel(uint32_t op, uint32_t nregs){\n')
f.write(f'if (op < {ops[0]} || op > {ops[-1]}) return 0;\n')
f.write(f'if (nregs < 1 || nregs > 16) return 0;\n')
f.write(f'return perf_kernels[(op - {ops[0]}) * 16 + (nregs - 1)];\n')
f.write('};\n')
