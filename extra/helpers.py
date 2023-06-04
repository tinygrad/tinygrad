from tinygrad.helpers import Timing

def enable_early_exec():
  import subprocess, multiprocessing
  qin: multiprocessing.Queue = multiprocessing.Queue()
  qout: multiprocessing.Queue = multiprocessing.Queue()
  def _early_exec_process(qin, qout):
    while 1:
      path, inp = qin.get()
      qout.put(subprocess.check_output(path, input=inp))
  p = multiprocessing.Process(target=_early_exec_process, args=(qin, qout))
  p.daemon = True
  p.start()
  def early_exec(x):
    qin.put(x)
    return qout.get()
  return early_exec

def proc(itermaker, q):
  for x in itermaker(): q.put(x)
  q.close()

def cross_process(itermaker, maxsize=8):
  # TODO: use cloudpickle for itermaker
  import multiprocessing
  q: multiprocessing.Queue = multiprocessing.Queue(maxsize)
  p = multiprocessing.Process(target=proc, args=(itermaker, q))
  p.daemon = True
  p.start()

  # TODO: write tests and handle exit case
  while 1: yield q.get()
