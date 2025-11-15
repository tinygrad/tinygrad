WARP_THREADS = 64
BASE_TILE_ROWS = 16
BASE_TILE_COLS = 16
BASE_TILE_NEPT = 4

def row_col(threadIdx_x):
  local_warpid = threadIdx_x // WARP_THREADS
  warp_laneid = threadIdx_x % WARP_THREADS

  ret = []

  for inner in range(BASE_TILE_NEPT):
    row = warp_laneid % 16
    col = 4 * (warp_laneid // 16)

    # row = 4 * (warp_laneid // 16)
    # col = warp_laneid % 16

    row_offset = 0
    col_offset = inner

    ret.append((row + row_offset, col + col_offset))

  return ret

if __name__ == "__main__":
  for threadIdx_x in range(64):
    row, col = zip(*row_col(threadIdx_x))
    print(f"Thread {threadIdx_x:2}: ", end="")
    for r, c in zip(row, col):
      print(f"({r:3},{c:3}) ", end="")
    print()

  unique_pairs = set()
  for threadIdx_x in range(64):
    rc_list = row_col(threadIdx_x)
    for rc in rc_list:
      unique_pairs.add(rc)
  assert len(unique_pairs) == 64 * BASE_TILE_NEPT, f"Expected {64 * BASE_TILE_NEPT} unique pairs, got {len(unique_pairs)}"

  # # Define the table structure strings
  # sep = "+--------+-------------+--------------+-------------+--------------+"
  # header = "| lane | v0.[15:0]   | v0.[31:16]   | v1.[15:0]   | v1.[31:16]   |"
  #
  # # Print the table header
  # print(sep)
  # print(header)
  # # Use a double-line separator like in the example
  # print(sep.replace("-", "="))
  #
  # # Loop through all threads
  # for threadIdx_x in range(64):
  #   # Get the list of (row, col) tuples for the current thread
  #   rc_list = row_col(threadIdx_x)
  #
  #   # Format the (row, col) tuples into "A[r][c]" strings
  #   # We expect exactly 4 tuples based on BASE_TILE_NEPT = 4
  #   val1 = f"A[{rc_list[0][0]}][{rc_list[0][1]}]"
  #   val2 = f"A[{rc_list[1][0]}][{rc_list[1][1]}]"
  #   val3 = f"A[{rc_list[2][0]}][{rc_list[2][1]}]"
  #   val4 = f"A[{rc_list[3][0]}][{rc_list[3][1]}]"
  #
  #   # Format the lane string for center-alignment in the 8-char column
  #   lane_str = f" {threadIdx_x}"
  #
  #   # Print the formatted data row
  #   # |{lane_str:^7}| -> 8 chars, e.g., "|  10   |"
  #   # | {val1: <11} | -> 13 chars, e.g., "| A[0][16]    |"
  #   # | {val2: <12} | -> 14 chars, e.g., "| A[0][17]     |"
  #   print(f"|{lane_str:^7}| {val1: <11} | {val2: <12} | {val3: <11} | {val4: <12} |")
  #   print(sep)
