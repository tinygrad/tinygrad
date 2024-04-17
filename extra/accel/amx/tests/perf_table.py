import csv
import sys

def md_table(test_name, z_spacing, field):
    values = {}
    max_num_z = 0
    max_num_threads = 0
    bytes_per_z = set()
    ops_unit = set()
    for row in rows:
        if row['test_name'] == test_name and row['z_spacing'] == z_spacing:
            bytes_per_z.add(int(row['bytes_per_z']))
            ops_unit.add(row['ops_unit'])
            num_z = int(row['num_z'])
            num_threads = int(row['num_threads'])
            max_num_z = max(max_num_z, num_z)
            max_num_threads = max(max_num_threads, num_threads)
            values[(num_z, num_threads)] = float(row[field]) / float(row['ns_elapsed'])
    bytes_per_z, = bytes_per_z
    ops_unit, = ops_unit
    if field == 'insn_total': ops_unit = 'insn/ns'
    header = ['Z Accumulators', '1 Thread'] + [f'{n} Threads' for n in range(2, max_num_threads+1)]
    print(f'{fname} {test_name} ({z_spacing} z)')
    print('|' + '|'.join(header) + '|')
    print('|' + '|'.join(['---:'] * (max_num_threads + 1)) + '|')
    for num_z in range(1, max_num_z + 1):
        data = [f'{num_z} ({num_z * bytes_per_z} bytes) per thread']
        for num_threads in range(1, max_num_threads + 1):
            value = values[(num_z, num_threads)]
            data.append('%.1f %s' % (value, ops_unit))
        print('|' + '|'.join(data) + '|')

def main():
    global fname, rows

    fname = sys.argv[1]
    test_name = sys.argv[2] if len(sys.argv) > 2 else ''

    rows = []
    with open(fname) as f:
        rows = list(csv.DictReader(f))

    test_names = {row['test_name'] for row in rows}
    if test_name not in test_names:
        matches = sorted({x for x in test_names if test_name in x})
        if not matches:
            print('No matching test names')
            exit(1)
        elif len(matches) > 1:
            print('Available test names:')
            for m in matches:
                print(f'  {m}')
            exit(1)
        else:
            test_name, = matches

    z_spacing = 'far'
    field = 'ops_total'
    for arg in sys.argv[3:]:
        if arg == 'near':
            z_spacing = 'near'
        elif arg == 'far':
            z_spacing = 'far'
        elif arg == 'ops':
            field = 'ops_total'
        elif arg == 'insn':
            field = 'insn_total'
        else:
            assert False, arg

    md_table(test_name, z_spacing, field)

if __name__ == '__main__':
    main()
