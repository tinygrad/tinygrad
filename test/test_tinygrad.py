# tests if tinygrad core is under 1000 lines
from pathlib import Path
import os
import unittest

TINYGRAD_PATH = Path(__file__).parent.parent / "tinygrad"

def count_lines(s):
    s = s.split('\n')
    return len(s)

def traverse_dir(files, base, count=0, traversed=[]):
    base = str(base)
    #files = [base+'/'+i for i in files]
    print(files, end='\n\n')
    #print(files)
    for i in files:
        if '__pycache__' in i or i in traversed:
            pass
        elif os.path.isfile(i):
            with open(i, 'r') as file:
                count += count_lines(file.read())
                traversed.append(i)
                #print(count)
        else:
            #print(i)
            count = traverse_dir([base + '/' + i.split('/')[-1] +'/' + j for j in os.listdir(i)], i, count)
            traversed.append(i)
        print(i)
        print(count, end='\n\n')
    return count

class TestTinygrad(unittest.TestCase):
    def test_tinygrad(self):
        line_count = traverse_dir([str(TINYGRAD_PATH)+'/'+i for i in os.listdir(TINYGRAD_PATH)], TINYGRAD_PATH)
        print(line_count)
        assert line_count <= 1000
if __name__ == "__main__":
    unittest.main()