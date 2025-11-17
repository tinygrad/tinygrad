#!/usr/bin/env python3
"""
Trace OLD winograd shapes to understand axis ordering
"""

print("OLD Winograd Shape Flow (from tensor.py:2554-2594)")
print("="*70)

# Example: bs=1, cin=16, cout=16, groups=1, input=64×64
bs, cin, cout, groups = 1, 16, 16, 1
HW = (3, 3)  # kernel size
HWI, HWO = (6, 6), (4, 4)  # winograd tile sizes
input_hw = (64, 64)
rcout = cout // groups  # 16

# After padding and pooling
tyx = tuple((dim + 0 + 0 - 2) // 4 for dim in input_hw)  # (16, 16) tiles
print(f"\nInput: bs={bs}, groups={groups}, cin={cin}, spatial={input_hw}")
print(f"Kernel: cout={cout}, rcout={rcout}")
print(f"Tiles: tyx={tyx} (spatial tiles)")

print(f"\n--- Step 1: Pool and permute ---")
d_shape_after_pool = (bs, groups, cin, *tyx, *HWI)
print(f"d after pool: {d_shape_after_pool}")
# = (1, 1, 16, 16, 16, 6, 6)

d_permuted = (*HWI, bs, groups, cin, *tyx)
print(f"d after permute to front: {d_permuted}")
# = (6, 6, 1, 1, 16, 16, 16)

print(f"\n--- Step 2: Weight reshape ---")
g_shape = (groups, rcout, cin, *HW)
print(f"g (weight): {g_shape}")
# = (1, 16, 16, 3, 3)

print(f"\n--- Step 3: Apply winograd transforms ---")
# _apply_winograd_matrix(Bt, d, 2) where d.shape = (6,6,1,1,16,16,16)
# Output shape: (6, 6, <rest of d's dims>)
dfactors_shape = (*HWI, bs, groups, 1, cin, *tyx)
print(f"dfactors = Bt @ d: {dfactors_shape}")
# = (6, 6, 1, 1, 1, 16, 16, 16)

# _apply_winograd_matrix(G, g, 2) where g.shape = (1,16,16,3,3)
# Output shape: (6, 6, <rest of g's dims>)
gfactors_shape = (*HWI, 1, groups, rcout, cin, *([1]*len(tyx)))
print(f"gfactors = G @ g: {gfactors_shape}")
# = (6, 6, 1, 1, 16, 16, 1, 1)

print(f"\n--- Step 4: Multiply and reduce ---")
product_shape = (*HWI, bs, groups, rcout, *tyx)
print(f"(gfactors * dfactors).sum(cin): {product_shape}")
# = (6, 6, 1, 1, 16, 16, 16)
print("                                 ^   ^   ^   ^    ^")
print("                                 6×6 bs grp cout tiles")

print(f"\n--- Step 5: Apply At transform ---")
# _apply_winograd_matrix(At, product, 2)
# At transforms first 2 dims: (6,6) → (4,4)
ret_shape = (*HWO, bs, groups, rcout, *tyx)
print(f"ret = At @ product: {ret_shape}")
# = (4, 4, 1, 1, 16, 16, 16)
print("                     ^   ^  ^  ^   ^    ^")
print("                     4×4 bs gr co  tiles")

print(f"\n--- Step 6: Permute and reshape to final output ---")
# Permute interleaves tyx and HWO
# Then reshape to (bs, cout, oy, ox)
final_shape = (bs, cout, tyx[0] * HWO[0], tyx[1] * HWO[1])
print(f"final output: {final_shape}")
# = (1, 16, 64, 64)

print(f"\n{'='*70}")
print("KEY OBSERVATION:")
print("="*70)
print("Before At transform, the shape is: (6, 6, bs, groups, cout, ty, tx)")
print("                                    ^   ^              ^      ^   ^")
print("                                    winograd dims    channel  spatial")
print("")
print("After bufferize, the loops would be organized as:")
print("  - Winograd dims (6×6) - inner loops or local work")
print("  - Batch/groups (1×1) - trivial")
print("  - Channels (16) - parallelizable")
print("  - Spatial tiles (16×16) - parallelizable")
print("")
print("The scheduler might create grid from: [ty, tx, cout] = [16, 16, 16]")
print("Or optimize to: [6, 6, 16] if it fuses some dimensions")
print("")
print("This is DIFFERENT from NEW which has: [16, 16, 16] then gets r_16_6_6")
