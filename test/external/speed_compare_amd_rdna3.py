import itertools
from tinygrad.extra.assembly.assembly_rdna import specialize_to_rdna, RDNALanguage, uops_to_rdna_asm  # Import added here
sys.path.append(os.path.abspath('../..'))

if __name__ == "__main__":
  lang = RDNALanguage()
  # Assuming function_name and uops are defined or obtained somewhere in your code
  specialized_asm, global_size, local_size, _ = uops_to_rdna_asm(function_name, uops)  # Function call updated
  print(specialized_asm, global_size, local_size)