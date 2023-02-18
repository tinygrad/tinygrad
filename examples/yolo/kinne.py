from tinygrad.tensor import Tensor
import numpy
import os

# Format Details:
#  A KINNE parameter set is stored as a set of files named "snoop_bin_*.bin",
#   where the * is a number starting at 0.
#  Each file is simply raw little-endian floats,
#   as readable by: numpy.fromfile(path, "<f4")
#   and as writable by: t.numpy().astype("<f4", "C").tofile(path)
# This format is intended to be extremely simple to get into literally anything.
# It is not intended to be structural or efficient - reloading a network when
#  unnecessary is inefficient anyway.
# Ultimately, the idea behind this is as a format that, while it will always
#  require code to implement, requires as little code as possible, and therefore
#  works as a suitable interchange for any situation.
# To add to the usability of the format, some informal metadata is provided,
#  in "meta.txt", which provides human-readable shape information.
# This is intended to help with debugging other implementations of the network,
#  by providing concrete human-readable information on tensor shapes.
# It is NOT meant to be read by machines.

class KinneDir:
  """
  A KinneDir is an intermediate object used to save or load a model.
  """

  def __init__(self, base: str, save: bool):
    """
    Opens a new KINNE directory with the given base path.
    If save is true, the directory is created if possible.
    (This does not create parents.)
    Save being true or false determines if tensors are loaded or saved.
    The base path is of the form "models/abc" - no trailing slash.
    It is important that if you wish to save in the current directory,
     you use ".", not the empty string.
    """
    if save and not os.path.isdir(base):
      os.mkdir(base)
    self.base = base + "/snoop_bin_"
    self.next_part_index = 0
    self.save = save
    if save:
      self.metadata = open(base + "/meta.txt", "w")

  def parameter(self, t: Tensor):
    """
    parameter loads or saves a parameter, given as a tensor.
    """
    path = f"{self.base}{self.next_part_index}.bin"
    if self.save:
      t.numpy().astype("<f4", "C").tofile(path)
      self.metadata.write(f"{self.next_part_index}: {t.shape}\n")
    else:
      t.assign(Tensor(numpy.fromfile(path, "<f4")).reshape(shape=t.shape))
    self.next_part_index += 1

  def parameters(self, params):
    """
    parameters loads or saves a sequence of parameters.
    It's intended for easily attaching to an existing model,
     assuming that your parameters list orders are consistent.
    (In other words, usage with tinygrad.utils.get_parameters isn't advised -
      it's too 'implicit'.)
    """
    for t in params:
      self.parameter(t)

  def close(self):
    if self.save:
      self.metadata.close()
