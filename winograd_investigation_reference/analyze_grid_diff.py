#!/usr/bin/env python3
"""
Analyze why grid organization differs and what it means for memory layout
"""

print("="*80)
print("GRID ORGANIZATION ANALYSIS")
print("="*80)

print("\nOLD: r_6_6_16_32_2_2_2_2")
print("  Global: gid.x=16, gid.y=6, gid.z=6")
print("  Local:  lid=32×2×2×2×2")
print("  → Parallelizes 16 on X axis first (channels)")

print("\nNEW: r_16_6_6_32_2_2_2_2")
print("  Global: gid.x=6, gid.y=6, gid.z=16")
print("  Local:  lid=32×2×2×2×2")
print("  → Parallelizes 6 on X axis first (spatial)")

print("\n" + "="*80)
print("MEMORY LAYOUT IMPACT")
print("="*80)

print("""
The grid organization affects:
1. Which threads access which memory locations
2. Whether memory writes are coalesced (consecutive) or scattered

OLD grid (r_6_6_16):
  - gid.x varies 0-15 (16 values) → channel dimension
  - Threads with consecutive gid.x write to nearby memory
  - Enables coalesced float2 writes

NEW grid (r_16_6_6):
  - gid.x varies 0-5 (6 values) → spatial dimension
  - gid.z varies 0-15 (16 values) → channel dimension
  - Threads with consecutive gid.x write to different channels
  - Results in scattered writes

To fix: Need to reorder axes so channel dimension comes first (X axis)
""")
