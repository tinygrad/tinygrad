lowered
c0 = UOp(Ops.PARAM, dtypes.float.ptr(256), (), 3)  -> 
c2 = UOp.special(256, 'lidx0', dtype=dtypes.int)
c4 = UOp(Ops.PARAM, dtypes.float.ptr(256), (), 0)
c6 = c4.index(c2, ptr=True).load()
c7 = UOp(Ops.PARAM, dtypes.float.ptr(256), (), 1)
c9 = c7.index(c2, ptr=True).load()
c11 = UOp(Ops.PARAM, dtypes.float.ptr(256), (), 2)
c13 = c11.index(c2, ptr=True).load()
c14 = c6*c9+c13
c15 = c0.index(c2, ptr=True).store(c14)
ast = c15.sink(arg=KernelInfo(name='fma', axis_types=(), dont_use_locals=False, applied_opts=(), opts_to_apply=None, estimates=None, beam=0)).rtag(1)




---

s_load_b64 s[2:3] , s[0:1], 0   -> s[2:3] gets &a
s_load_b64 s[4:5] , s[0:1], 8 	-> &b
s_load_b64 s[6:7] , s[0:1], 16  -> &c 
s_load_b64 s[8:9] , s[0:1], 24 -> &out


Indexing -> we know that each element is 4 bytes. So each thread gets to load its tid * 4 + base.
so for instance, lets say &a is 0x1000
tid 0 -> read(0x1000)
1 -> read(0x0004)
2 -> read(0x1008)
3 -> read(0x1012)

naturally coalesc3ed :0

so the natural way to do this is

v_lshlrev_b32(v0, 2, v0) vector left shift left reverse operand order -> vdst, srd0:u32, vscr1 so v0 = v0 << 2

now we have the v0 expressed as the offset we use to index into a buff3er


so now we load ->
global_load_b32 v1, v0, s[2:3]   #we want v1 = a[tid]
global_load_b32 v2, v0, s[4:5]
global_load_b32 v3, v0, s[6:7]

// now we stall to wait
s_waitcnt vmcnt(0) 

v_mul_f32 v1, v1, v2 -> a = a * b
v_add_f32 v1,v1,v3 -> a = a + c

global_store_b32 (v0, v1, s[8:9] 

s_endpgm
